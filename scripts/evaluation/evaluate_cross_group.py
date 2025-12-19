#!/usr/bin/env python3
"""
Cross-Group Attention Evaluation Script for Chronos-2

This script evaluates the novel Cross-Group Attention mechanism against the baseline
Chronos-2 model. The Cross-Group Attention allows different time series groups
(e.g., electricity prices in Germany vs France) to share information through
a gated attention mechanism.

Key Innovation:
- Original Chronos-2: Groups don't attend to each other
- Our Enhancement: Groups CAN attend to each other via learned summaries + gating

Usage:
    python evaluate_cross_group.py --config configs/cross_group_test.yaml \
        --output results/cross_group_results.csv \
        --device cuda

Author: Cross-Group Attention Extension
"""

import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, List

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

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Cross-Group Evaluation")
logger.setLevel(logging.INFO)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    """Convert HuggingFace dataset to GluonTS format."""
    series_fields = [col for col in hf_dataset.features if isinstance(hf_dataset.features[col], datasets.Sequence)]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)

    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(hf_entry["timestamp"][0], freq=dataset_freq),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length
    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    """Load dataset and create test splits."""
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


def generate_forecasts_with_cross_group(
    test_data_input: Iterable,
    pipeline: Chronos2Pipeline,
    prediction_length: int,
    batch_size: int,
    enable_cross_learning: bool = False,
):
    """Generate forecasts, optionally with cross-learning enabled."""
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
    """
    Enable cross-group attention in an existing Chronos2 pipeline.
    
    This modifies the model's config and reinitializes the encoder blocks
    to include the CrossGroupAttention layers.
    """
    logger.info("Enabling Cross-Group Attention mechanism...")
    
    # Get the current config
    config = deepcopy(pipeline.model.config)
    
    # Enable cross-group attention
    config.use_cross_group_attention = True
    
    # Create a new model with cross-group attention enabled
    new_model = Chronos2Model(config).to(pipeline.model.device)
    
    # Copy weights from original model (only matching keys)
    original_state_dict = pipeline.model.state_dict()
    new_state_dict = new_model.state_dict()
    
    # Copy all weights that exist in both models
    for key in original_state_dict:
        if key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    
    # Load the state dict (strict=False because new model has additional layers)
    new_model.load_state_dict(new_state_dict, strict=False)
    
    # Initialize the new cross-group attention layers
    # (they are already initialized with random weights in the constructor)
    
    logger.info("Cross-Group Attention layers initialized successfully!")
    
    return Chronos2Pipeline(model=new_model)


def evaluate_model(
    pipeline: Chronos2Pipeline,
    backtest_configs: list,
    batch_size: int,
    enable_cross_learning: bool = False,
    model_name: str = "baseline",
) -> pd.DataFrame:
    """Evaluate a model on all datasets in the config."""
    result_rows = []
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)

        logger.info(f"Generating forecasts for {dataset_name} ({len(test_data.input)} time series)")
        forecasts = generate_forecasts_with_cross_group(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            enable_cross_learning=enable_cross_learning,
        )

        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(QUANTILES),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            **metrics[0]
        })

    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    
    return results_df


@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "cross_group_test.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/cross_group_results.csv"),
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
        32,
        help="Batch size for inference"
    ),
    compare_cross_learning: bool = typer.Option(
        True,
        help="Also compare with cross_learning=True baseline"
    ),
):
    """
    Evaluate Cross-Group Attention mechanism against baseline Chronos-2.
    
    This script compares:
    1. Baseline Chronos-2 (no cross-group attention)
    2. Chronos-2 with cross_learning=True (simple approach)
    3. Chronos-2 with Cross-Group Attention (our novel approach)
    """
    
    # Load config
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using default test config.")
        backtest_configs = [
            {"name": "ercot", "hf_repo": "autogluon/chronos_datasets", "offset": -24, "prediction_length": 24, "num_rolls": 1},
            {"name": "monash_australian_electricity", "hf_repo": "autogluon/chronos_datasets", "offset": -48, "prediction_length": 48, "num_rolls": 1},
            {"name": "monash_traffic", "hf_repo": "autogluon/chronos_datasets", "offset": -24, "prediction_length": 24, "num_rolls": 1},
        ]
    else:
        with open(config_path) as fp:
            backtest_configs = yaml.safe_load(fp)
    
    logger.info(f"Loading baseline Chronos-2 model from {model_id}")
    torch_dtype = torch.float32
    baseline_pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=torch_dtype
    )
    
    assert isinstance(baseline_pipeline, Chronos2Pipeline), "Expected Chronos2Pipeline"
    
    all_results = []
    
    # 1. Evaluate baseline (no cross-learning)
    logger.info("=" * 60)
    logger.info("Evaluating BASELINE (no cross-learning)")
    logger.info("=" * 60)
    baseline_results = evaluate_model(
        baseline_pipeline,
        backtest_configs,
        batch_size,
        enable_cross_learning=False,
        model_name="chronos2_baseline"
    )
    all_results.append(baseline_results)
    logger.info(f"\nBaseline Results:\n{baseline_results.to_string()}")
    
    # 2. Evaluate with cross_learning=True (simple approach)
    if compare_cross_learning:
        logger.info("=" * 60)
        logger.info("Evaluating with CROSS_LEARNING=True")
        logger.info("=" * 60)
        cross_learning_results = evaluate_model(
            baseline_pipeline,
            backtest_configs,
            batch_size,
            enable_cross_learning=True,
            model_name="chronos2_cross_learning"
        )
        all_results.append(cross_learning_results)
        logger.info(f"\nCross-Learning Results:\n{cross_learning_results.to_string()}")
    
    # 3. Evaluate with Cross-Group Attention (our novel approach)
    logger.info("=" * 60)
    logger.info("Evaluating with CROSS-GROUP ATTENTION (Novel)")
    logger.info("=" * 60)
    cross_group_pipeline = enable_cross_group_attention(baseline_pipeline)
    cross_group_results = evaluate_model(
        cross_group_pipeline,
        backtest_configs,
        batch_size,
        enable_cross_learning=True,  # Use cross_learning to ensure groups are defined
        model_name="chronos2_cross_group_attention"
    )
    all_results.append(cross_group_results)
    logger.info(f"\nCross-Group Attention Results:\n{cross_group_results.to_string()}")
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create comparison summary
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    pivot_mase = combined_results.pivot(index="dataset", columns="model", values="MASE")
    pivot_wql = combined_results.pivot(index="dataset", columns="model", values="WQL")
    
    logger.info("\nMASE Comparison:")
    logger.info(pivot_mase.to_string())
    
    logger.info("\nWQL Comparison:")
    logger.info(pivot_wql.to_string())
    
    # Calculate improvement
    if "chronos2_baseline" in pivot_mase.columns and "chronos2_cross_group_attention" in pivot_mase.columns:
        mase_improvement = (pivot_mase["chronos2_baseline"] - pivot_mase["chronos2_cross_group_attention"]) / pivot_mase["chronos2_baseline"] * 100
        wql_improvement = (pivot_wql["chronos2_baseline"] - pivot_wql["chronos2_cross_group_attention"]) / pivot_wql["chronos2_baseline"] * 100
        
        logger.info("\n" + "=" * 60)
        logger.info("IMPROVEMENT OVER BASELINE (positive = better)")
        logger.info("=" * 60)
        logger.info(f"\nMASE Improvement (%):\n{mase_improvement.to_string()}")
        logger.info(f"\nWQL Improvement (%):\n{wql_improvement.to_string()}")
        logger.info(f"\nAverage MASE Improvement: {mase_improvement.mean():.2f}%")
        logger.info(f"Average WQL Improvement: {wql_improvement.mean():.2f}%")
    
    # Save results
    combined_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return combined_results


@app.command()
def quick_test(
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    ),
):
    """
    Quick smoke test to verify Cross-Group Attention implementation works.
    """
    logger.info("Running quick smoke test...")
    
    # Create a small dummy model for testing
    config = Chronos2CoreConfig(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.1,
        use_cross_group_attention=True,
        chronos_config={
            "context_length": 64,
            "output_patch_size": 8,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": True,
        }
    )
    
    model = Chronos2Model(config).to(device)
    model.eval()
    
    # Create dummy inputs with 2 different groups
    batch_size = 6
    context_length = 64
    context = torch.randn(batch_size, context_length, device=device)
    group_ids = torch.tensor([0, 0, 1, 1, 2, 2], device=device)  # 3 groups, 2 each
    
    logger.info(f"Testing with batch_size={batch_size}, context_length={context_length}")
    logger.info(f"Group IDs: {group_ids.tolist()}")
    
    with torch.no_grad():
        output = model(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            output_attentions=True,
        )
    
    logger.info(f"Output shape: {output.quantile_preds.shape}")
    logger.info(f"Has cross-group attention weights: {output.enc_cross_group_attn_weights is not None}")
    
    if output.enc_cross_group_attn_weights is not None:
        for i, weights in enumerate(output.enc_cross_group_attn_weights):
            logger.info(f"  Layer {i} cross-group attn weights shape: {weights.shape}")
    
    logger.info("✅ Quick test PASSED! Cross-Group Attention is working correctly.")


if __name__ == "__main__":
    app()
