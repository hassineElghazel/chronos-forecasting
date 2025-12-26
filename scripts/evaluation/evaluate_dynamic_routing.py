#!/usr/bin/env python3
"""
Dynamic Routing Evaluation Script

Tests cross-group attention ALONE (no cross_learning interference) with dynamic routing.
Tracks when cross-group attention is applied vs skipped based on group similarity.

Key output:
- Per-dataset similarity statistics
- When cross-group attention was applied (high similarity) vs skipped (low similarity)
- Performance comparison: baseline vs cross_group_dynamic
"""

import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

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
logger = logging.getLogger("Dynamic Routing Eval")
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


def get_cross_group_stats(model: Chronos2Model) -> dict:
    """Extract cross-group attention statistics from model."""
    stats = {
        "total_batches": 0,
        "applied_count": 0,
        "skipped_count": 0,
        "skipped_low_sim": 0,
        "skipped_low_margin": 0,
        "avg_similarities": [],
        "margins": [],
        "applied_similarities": [],
        "applied_margins": [],
        "skipped_similarities": [],
        "skipped_margins": [],
    }
    
    # Iterate through encoder blocks and collect stats (note: it's .block not .blocks)
    for block in model.encoder.block:
        if hasattr(block, 'cross_group_attention') and block.cross_group_attention is not None:
            cga = block.cross_group_attention
            if hasattr(cga, '_stats_history'):
                for entry in cga._stats_history:
                    # Handle both old format (avg_sim, was_applied) and new format (avg_sim, margin, was_applied, reason)
                    if len(entry) == 2:
                        avg_sim, was_applied = entry
                        margin, reason = 0.0, "applied" if was_applied else "skipped"
                    else:
                        avg_sim, margin, was_applied, reason = entry
                    
                    stats["total_batches"] += 1
                    stats["avg_similarities"].append(avg_sim)
                    stats["margins"].append(margin)
                    
                    if was_applied:
                        stats["applied_count"] += 1
                        stats["applied_similarities"].append(avg_sim)
                        stats["applied_margins"].append(margin)
                    else:
                        stats["skipped_count"] += 1
                        stats["skipped_similarities"].append(avg_sim)
                        stats["skipped_margins"].append(margin)
                        if reason == "low_avg_sim":
                            stats["skipped_low_sim"] += 1
                        elif reason == "low_margin":
                            stats["skipped_low_margin"] += 1
    
    return stats


def reset_cross_group_stats(model: Chronos2Model):
    """Reset cross-group attention statistics."""
    for block in model.encoder.block:
        if hasattr(block, 'cross_group_attention') and block.cross_group_attention is not None:
            cga = block.cross_group_attention
            if hasattr(cga, '_stats_history'):
                cga._stats_history = []


def generate_forecasts(
    test_data_input,
    pipeline: Chronos2Pipeline,
    prediction_length: int,
    batch_size: int,
):
    """Generate forecasts WITHOUT cross_learning (each series is its own group)."""
    forecast_outputs = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Generating forecasts"):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=QUANTILES,
            cross_learning=False,  # IMPORTANT: No cross_learning, let cross_group work alone
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


def enable_cross_group_dynamic(
    pipeline: Chronos2Pipeline,
    dynamic_threshold: float = 0.5,
    margin_gate: bool = False,
    margin_delta: float = 0.1,
) -> Chronos2Pipeline:
    """Enable cross-group attention with dynamic routing and optional margin gate."""
    gate_str = f", margin_gate={margin_gate}, margin_delta={margin_delta}" if margin_gate else ""
    logger.info(f"Enabling Cross-Group Attention with dynamic routing (threshold={dynamic_threshold}{gate_str})")
    config = deepcopy(pipeline.model.config)
    config.use_cross_group_attention = True
    config.cross_group_dynamic_routing = True
    config.cross_group_dynamic_threshold = dynamic_threshold
    config.cross_group_margin_gate = margin_gate
    config.cross_group_margin_delta = margin_delta
    
    new_model = Chronos2Model(config).to(pipeline.model.device)
    original_state_dict = pipeline.model.state_dict()
    new_state_dict = new_model.state_dict()
    for key in original_state_dict:
        if key in new_state_dict:
            new_state_dict[key] = original_state_dict[key]
    new_model.load_state_dict(new_state_dict, strict=False)
    logger.info("Cross-Group Attention with dynamic routing initialized!")
    return Chronos2Pipeline(model=new_model)


def evaluate_single_config(
    pipeline: Chronos2Pipeline,
    backtest_configs: list,
    batch_size: int,
    model_name: str,
    track_stats: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate a single configuration across all datasets."""
    result_rows = []
    all_stats = {}
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info(f"  [{model_name}] Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)
        num_series = len(test_data.input)
        
        # Reset stats before evaluation
        if track_stats:
            reset_cross_group_stats(pipeline.model)
        
        start_time = time.time()
        logger.info(f"  [{model_name}] Generating forecasts for {dataset_name} ({num_series} series)")
        
        forecasts = generate_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
        )
        
        # Collect stats after evaluation
        if track_stats:
            stats = get_cross_group_stats(pipeline.model)
            all_stats[dataset_name] = stats
            
            # Log stats
            if stats["total_batches"] > 0:
                applied_pct = stats["applied_count"] / stats["total_batches"] * 100
                avg_sim = np.mean(stats["avg_similarities"]) if stats["avg_similarities"] else 0
                logger.info(f"  [{model_name}] {dataset_name} STATS: "
                           f"applied={applied_pct:.1f}% ({stats['applied_count']}/{stats['total_batches']}), "
                           f"avg_similarity={avg_sim:.3f}")
        
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
    
    df = pd.DataFrame(result_rows).rename(
        {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
        axis="columns",
    )
    return df, all_stats


@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "no_dominick.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/dynamic_routing_results.csv"),
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
    dynamic_threshold: float = typer.Option(
        0.5,
        help="Similarity threshold for dynamic routing"
    ),
    margin_gate: bool = typer.Option(
        False,
        help="Enable margin gate (require top1-median > delta)"
    ),
    margin_delta: float = typer.Option(
        0.1,
        help="Min margin between top1 and median similarity"
    ),
):
    """
    Evaluate cross-group attention ALONE with dynamic routing.
    
    Tests:
    1. Baseline (no cross_group, no cross_learning)
    2. Cross-group with dynamic routing (skips if groups are dissimilar)
    
    Optionally enable margin gate to reduce false positives.
    Tracks when cross-group attention is applied vs skipped.
    """
    
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
    
    # 1. BASELINE
    logger.info("=" * 60)
    logger.info("CONFIG 1: BASELINE (no cross_group, no cross_learning)")
    logger.info("=" * 60)
    results, _ = evaluate_single_config(
        baseline_pipeline, backtest_configs, batch_size,
        model_name="baseline", track_stats=False
    )
    all_results.append(results)
    
    # 2. CROSS-GROUP DYNAMIC
    logger.info("=" * 60)
    gate_str = f"+margin({margin_delta})" if margin_gate else ""
    logger.info(f"CONFIG 2: CROSS-GROUP DYNAMIC (threshold={dynamic_threshold}{gate_str})")
    logger.info("=" * 60)
    dynamic_pipeline = enable_cross_group_dynamic(
        baseline_pipeline, dynamic_threshold, margin_gate, margin_delta
    )
    model_name = f"cga_dyn_{dynamic_threshold}"
    if margin_gate:
        model_name += f"_margin{margin_delta}"
    results, stats = evaluate_single_config(
        dynamic_pipeline, backtest_configs, batch_size,
        model_name=model_name, track_stats=True
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
    
    # Cross-group application statistics
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-GROUP ATTENTION APPLICATION STATISTICS")
    logger.info("=" * 60)
    
    for dataset_name, dataset_stats in stats.items():
        if dataset_stats["total_batches"] > 0:
            applied_pct = dataset_stats["applied_count"] / dataset_stats["total_batches"] * 100
            avg_sim = np.mean(dataset_stats["avg_similarities"]) if dataset_stats["avg_similarities"] else 0
            avg_margin = np.mean(dataset_stats["margins"]) if dataset_stats["margins"] else 0
            applied_avg = np.mean(dataset_stats["applied_similarities"]) if dataset_stats["applied_similarities"] else 0
            skipped_avg = np.mean(dataset_stats["skipped_similarities"]) if dataset_stats["skipped_similarities"] else 0
            
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Total batches: {dataset_stats['total_batches']}")
            logger.info(f"  Applied: {dataset_stats['applied_count']} ({applied_pct:.1f}%)")
            logger.info(f"  Skipped: {dataset_stats['skipped_count']} ({100-applied_pct:.1f}%)")
            if dataset_stats.get("skipped_low_sim", 0) > 0 or dataset_stats.get("skipped_low_margin", 0) > 0:
                logger.info(f"    - skipped (low_sim): {dataset_stats.get('skipped_low_sim', 0)}")
                logger.info(f"    - skipped (low_margin): {dataset_stats.get('skipped_low_margin', 0)}")
            logger.info(f"  Avg similarity (all): {avg_sim:.4f}")
            logger.info(f"  Avg margin (all): {avg_margin:.4f}")
            logger.info(f"  Avg similarity (when applied): {applied_avg:.4f}")
            logger.info(f"  Avg similarity (when skipped): {skipped_avg:.4f}")
    
    # Improvement analysis
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVEMENT OVER BASELINE (positive = better)")
    logger.info("=" * 60)
    
    # Find the model column (not baseline)
    model_cols = [c for c in pivot_mase.columns if c != "baseline"]
    if model_cols:
        model_col = model_cols[0]
        for ds in pivot_mase.index:
            mase_imp = (pivot_mase.loc[ds, "baseline"] - pivot_mase.loc[ds, model_col]) / pivot_mase.loc[ds, "baseline"] * 100
            wql_imp = (pivot_wql.loc[ds, "baseline"] - pivot_wql.loc[ds, model_col]) / pivot_wql.loc[ds, "baseline"] * 100
            
            # Get application rate for this dataset
            ds_stats = stats.get(ds, {})
            applied_pct = ds_stats.get("applied_count", 0) / max(ds_stats.get("total_batches", 1), 1) * 100
            avg_sim = np.mean(ds_stats.get("avg_similarities", [0]))
            avg_margin = np.mean(ds_stats.get("margins", [0]))
            
            logger.info(f"  {ds}: MASE={mase_imp:+.2f}%, WQL={wql_imp:+.2f}% "
                       f"[applied={applied_pct:.0f}%, sim={avg_sim:.3f}, margin={avg_margin:.3f}]")
    
    # Save
    combined.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return combined


if __name__ == "__main__":
    app()
