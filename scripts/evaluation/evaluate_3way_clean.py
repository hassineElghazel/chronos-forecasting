#!/usr/bin/env python3
"""
3-Way Clean Comparison

Compares:
1. baseline - no cross_group, no cross_learning
2. cross_learning - cross_learning only (token-level sharing)
3. cga_dynamic - cross_group with dynamic routing (threshold=0.9), no cross_learning
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
from chronos.chronos2.model import Chronos2Model

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("3-Way Comparison")
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
    cross_learning: bool = False,
):
    """Generate forecasts with optional cross_learning."""
    forecast_outputs = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Forecasting"):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=QUANTILES,
            cross_learning=cross_learning,
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


def enable_cross_group_dynamic(pipeline: Chronos2Pipeline, threshold: float = 0.9) -> Chronos2Pipeline:
    """Enable cross-group attention with dynamic routing."""
    config = deepcopy(pipeline.model.config)
    config.use_cross_group_attention = True
    config.cross_group_dynamic_routing = True
    config.cross_group_dynamic_threshold = threshold
    
    new_model = Chronos2Model(config).to(pipeline.model.device)
    original_state_dict = pipeline.model.state_dict()
    new_state_dict = new_model.state_dict()
    for key in original_state_dict:
        if key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    new_model.load_state_dict(new_state_dict, strict=False)
    return Chronos2Pipeline(model=new_model)


def evaluate_config(
    pipeline: Chronos2Pipeline,
    backtest_configs: list,
    batch_size: int,
    model_name: str,
    cross_learning: bool = False,
) -> pd.DataFrame:
    """Evaluate a single configuration across all datasets."""
    result_rows = []
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info(f"  [{model_name}] {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)
        
        start_time = time.time()
        forecasts = generate_forecasts(
            test_data.input, pipeline, prediction_length, batch_size, cross_learning
        )
        
        metrics = (
            evaluate_forecasts(
                forecasts, test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        
        wall_time = time.time() - start_time
        mase = metrics[0].get("MASE[0.5]", 0.0)
        wql = metrics[0].get("mean_weighted_sum_quantile_loss", 0.0)
        
        logger.info(f"    MASE={mase:.4f}, WQL={wql:.4f} ({wall_time:.1f}s)")
        
        result_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            **metrics[0]
        })
    
    return pd.DataFrame(result_rows).rename(
        {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"}, axis="columns"
    )


@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "no_dominick.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/3way_clean_results.csv"),
        help="Path to save the results CSV"
    ),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = typer.Option(8),
    dynamic_threshold: float = typer.Option(0.9),
):
    """
    3-way comparison:
    1. baseline (no cross_learning, no CGA)
    2. cross_learning (token-level sharing)
    3. cga_dynamic (group-level sharing with dynamic routing)
    """
    
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    
    logger.info(f"Loading model from {model_id}")
    baseline_pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.float32
    )
    
    all_results = []
    
    # 1. BASELINE
    logger.info("=" * 50)
    logger.info("CONFIG 1: BASELINE")
    logger.info("=" * 50)
    results = evaluate_config(baseline_pipeline, backtest_configs, batch_size, "baseline", cross_learning=False)
    all_results.append(results)
    
    # 2. CROSS_LEARNING only
    logger.info("=" * 50)
    logger.info("CONFIG 2: CROSS_LEARNING")
    logger.info("=" * 50)
    results = evaluate_config(baseline_pipeline, backtest_configs, batch_size, "cross_learning", cross_learning=True)
    all_results.append(results)
    
    # 3. CGA_DYNAMIC only (no cross_learning)
    logger.info("=" * 50)
    logger.info(f"CONFIG 3: CGA_DYNAMIC (threshold={dynamic_threshold})")
    logger.info("=" * 50)
    cga_pipeline = enable_cross_group_dynamic(baseline_pipeline, dynamic_threshold)
    results = evaluate_config(cga_pipeline, backtest_configs, batch_size, "cga_dynamic", cross_learning=False)
    all_results.append(results)
    
    # Combine and analyze
    combined = pd.concat(all_results, ignore_index=True)
    
    # Summary tables
    logger.info("\n" + "=" * 60)
    logger.info("MASE RESULTS")
    logger.info("=" * 60)
    pivot_mase = combined.pivot(index="dataset", columns="model", values="MASE")
    pivot_mase = pivot_mase[["baseline", "cross_learning", "cga_dynamic"]]
    logger.info("\n" + pivot_mase.to_string())
    
    logger.info("\n" + "=" * 60)
    logger.info("WQL RESULTS")
    logger.info("=" * 60)
    pivot_wql = combined.pivot(index="dataset", columns="model", values="WQL")
    pivot_wql = pivot_wql[["baseline", "cross_learning", "cga_dynamic"]]
    logger.info("\n" + pivot_wql.to_string())
    
    # Improvement over baseline
    logger.info("\n" + "=" * 60)
    logger.info("MASE IMPROVEMENT OVER BASELINE (positive = better)")
    logger.info("=" * 60)
    for model in ["cross_learning", "cga_dynamic"]:
        imp = (pivot_mase["baseline"] - pivot_mase[model]) / pivot_mase["baseline"] * 100
        avg_imp = imp.mean()
        logger.info(f"\n{model}:")
        for ds in imp.index:
            logger.info(f"  {ds}: {imp[ds]:+.2f}%")
        logger.info(f"  AVERAGE: {avg_imp:+.2f}%")
    
    # Best model per dataset
    logger.info("\n" + "=" * 60)
    logger.info("BEST MODEL PER DATASET (MASE)")
    logger.info("=" * 60)
    for ds in pivot_mase.index:
        best = pivot_mase.loc[ds].idxmin()
        best_val = pivot_mase.loc[ds].min()
        logger.info(f"  {ds}: {best} ({best_val:.4f})")
    
    combined.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return combined


if __name__ == "__main__":
    app()
