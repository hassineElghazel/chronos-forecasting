#!/usr/bin/env python3
"""
4-Way Comparison Evaluation Script

Tests:
1. Baseline (no cross_learning, no cross_group_attention)
2. Cross-learning alone (cross_learning=True, no cross_group_attention)
3. Cross-group attention alone (cross_learning=False, cross_group_attention=True)
4. Combined (cross_learning=True + cross_group_attention=True)

This validates whether the combination provides additional benefit.
"""

import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("4-Way Evaluation")
logger.setLevel(logging.INFO)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [col for col in hf_dataset.features if isinstance(hf_dataset.features[col], datasets.Sequence)]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr
    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append({
                "start": pd.Period(hf_entry["timestamp"][0], freq=dataset_freq),
                "target": hf_entry[field],
            })
    assert len(gts_dataset) == dataset_length
    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False
    ds = datasets.load_dataset(hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code)
    ds.set_format("numpy")
    gts_dataset = to_gluonts_univariate(ds)
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)
    return test_data


def generate_forecasts(
    test_data_input,
    pipeline: Chronos2Pipeline,
    prediction_length: int,
    batch_size: int,
    enable_cross_learning: bool = False,
):
    forecast_outputs = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Generating forecasts"):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=QUANTILES,
            cross_learning=enable_cross_learning,
        )
        if isinstance(quantiles, list):
            quantiles = np.stack(quantiles).squeeze(axis=1)
        quantiles = quantiles.swapaxes(-1, -2)
        forecast_outputs.append(quantiles)
    forecast_outputs = np.concatenate(forecast_outputs)

    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        forecasts.append(
            QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, QUANTILES)),
                start_date=forecast_start_date,
            )
        )
    return forecasts


def enable_cross_group_attention(pipeline: Chronos2Pipeline) -> Chronos2Pipeline:
    """Enable cross-group attention (full, no selective masking)."""
    logger.info("Enabling Cross-Group Attention...")
    config = deepcopy(pipeline.model.config)
    config.use_cross_group_attention = True
    new_model = Chronos2Model(config).to(pipeline.model.device)
    original_state_dict = pipeline.model.state_dict()
    new_state_dict = new_model.state_dict()
    for key in original_state_dict:
        if key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    new_model.load_state_dict(new_state_dict, strict=False)
    logger.info("Cross-Group Attention layers initialized!")
    return Chronos2Pipeline(model=new_model)


def evaluate_single_config(
    pipeline: Chronos2Pipeline,
    backtest_configs: list,
    batch_size: int,
    enable_cross_learning: bool,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate a single configuration across all datasets."""
    result_rows = []
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info(f"  [{model_name}] Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)
        num_series = len(test_data.input)
        
        start_time = time.time()
        logger.info(f"  [{model_name}] Generating forecasts for {dataset_name} ({num_series} series)")
        
        forecasts = generate_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            enable_cross_learning=enable_cross_learning,
        )
        
        logger.info(f"  [{model_name}] Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        
        wall_time = time.time() - start_time
        mase = metrics[0].get("MASE[0.5]", 0.0)
        wql = metrics[0].get("mean_weighted_sum_quantile_loss", 0.0)
        
        logger.info(f"  [{model_name}] {dataset_name}: MASE={mase:.4f}, WQL={wql:.4f} ({wall_time:.1f}s)")
        
        result_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            **metrics[0]
        })
    
    return pd.DataFrame(result_rows).rename(
        {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
        axis="columns",
    )


@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "no_dominick.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/4way_results.csv"),
        help="Path to save the results CSV"
    ),
    model_id: str = typer.Option(
        "amazon/chronos-2",
        help="HuggingFace ID of the Chronos-2 model"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    ),
    batch_size: int = typer.Option(
        8,
        help="Batch size for inference"
    ),
):
    """
    Run 4-way comparison:
    1. Baseline (no cross_learning, no cross_group)
    2. Cross-learning alone
    3. Cross-group attention alone  
    4. Combined (cross_learning + cross_group)
    """
    
    # Load config
    if not config_path.exists():
        logger.error(f"Config file {config_path} not found!")
        return
    
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    
    logger.info(f"Loading Chronos-2 model from {model_id}")
    torch_dtype = torch.float32
    baseline_pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=torch_dtype
    )
    assert isinstance(baseline_pipeline, Chronos2Pipeline)
    
    all_results = []
    
    # 1. BASELINE: no cross_learning, no cross_group
    logger.info("=" * 60)
    logger.info("CONFIG 1: BASELINE (no cross_learning, no cross_group)")
    logger.info("=" * 60)
    results = evaluate_single_config(
        baseline_pipeline, backtest_configs, batch_size,
        enable_cross_learning=False, model_name="baseline"
    )
    all_results.append(results)
    
    # 2. CROSS-LEARNING ALONE: cross_learning=True, no cross_group
    logger.info("=" * 60)
    logger.info("CONFIG 2: CROSS-LEARNING ALONE")
    logger.info("=" * 60)
    results = evaluate_single_config(
        baseline_pipeline, backtest_configs, batch_size,
        enable_cross_learning=True, model_name="cross_learning"
    )
    all_results.append(results)
    
    # 3. CROSS-GROUP ALONE: cross_learning=False, cross_group=True
    logger.info("=" * 60)
    logger.info("CONFIG 3: CROSS-GROUP ATTENTION ALONE")
    logger.info("=" * 60)
    cga_pipeline = enable_cross_group_attention(baseline_pipeline)
    results = evaluate_single_config(
        cga_pipeline, backtest_configs, batch_size,
        enable_cross_learning=False, model_name="cross_group_alone"
    )
    all_results.append(results)
    
    # 4. COMBINED: cross_learning=True + cross_group=True
    logger.info("=" * 60)
    logger.info("CONFIG 4: COMBINED (cross_learning + cross_group)")
    logger.info("=" * 60)
    results = evaluate_single_config(
        cga_pipeline, backtest_configs, batch_size,
        enable_cross_learning=True, model_name="combined"
    )
    all_results.append(results)
    
    # Combine and analyze
    combined = pd.concat(all_results, ignore_index=True)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    pivot_mase = combined.pivot(index="dataset", columns="model", values="MASE")
    pivot_wql = combined.pivot(index="dataset", columns="model", values="WQL")
    
    logger.info("\nMASE by model:")
    logger.info(pivot_mase.to_string())
    
    logger.info("\nWQL by model:")
    logger.info(pivot_wql.to_string())
    
    # Improvement over baseline
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVEMENT OVER BASELINE (positive = better)")
    logger.info("=" * 60)
    
    for model in ["cross_learning", "cross_group_alone", "combined"]:
        if model in pivot_mase.columns:
            mase_imp = ((pivot_mase["baseline"] - pivot_mase[model]) / pivot_mase["baseline"] * 100)
            wql_imp = ((pivot_wql["baseline"] - pivot_wql[model]) / pivot_wql["baseline"] * 100)
            logger.info(f"\n{model}:")
            for ds in pivot_mase.index:
                logger.info(f"  {ds}: MASE={mase_imp[ds]:+.2f}%, WQL={wql_imp[ds]:+.2f}%")
            logger.info(f"  AVERAGE: MASE={mase_imp.mean():+.2f}%, WQL={wql_imp.mean():+.2f}%")
    
    # Save
    combined.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return combined


if __name__ == "__main__":
    app()
