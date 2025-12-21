#!/usr/bin/env python3
"""
Cross-Group Attention Evaluation Script for Chronos-2

This script evaluates the novel Selective Cross-Group Attention mechanism against
the baseline Chronos-2 model. The Cross-Group Attention allows different time series
groups (e.g., electricity prices in Germany vs France) to share information through
a gated attention mechanism with selective sharing strategies.

Key Innovations:
- Original Chronos-2: Groups don't attend to each other
- Cross-Group Attention: Groups CAN attend to each other via learned summaries + gating
- Selective Mechanisms: Top-k, similarity threshold, or sparse routing to reduce negative transfer

Usage:
    python evaluate_cross_group.py evaluate --config configs/cross_group_test.yaml \
        --output results/cross_group_results.csv \
        --device cuda \
        --cross-group-top-k 3

    python evaluate_cross_group.py ablation --config configs/cross_group_extended.yaml \
        --output results/ablation_results.csv

Author: Cross-Group Attention Extension
"""

import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

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


# =============================================================================
# Memory & Complexity Instrumentation
# =============================================================================

@dataclass
class MemoryStats:
    """Container for memory statistics."""
    device_type: str  # "cuda", "mps", or "cpu"
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_peak_memory_allocated_mb: float = 0.0
    gpu_peak_memory_reserved_mb: float = 0.0
    cpu_rss_mb: float = 0.0
    
    def to_log_string(self) -> str:
        """Format as single-line key=value string for easy parsing."""
        if self.device_type == "cuda":
            return (f"device={self.device_type} "
                    f"gpu_alloc_mb={self.gpu_memory_allocated_mb:.2f} "
                    f"gpu_reserved_mb={self.gpu_memory_reserved_mb:.2f} "
                    f"gpu_peak_alloc_mb={self.gpu_peak_memory_allocated_mb:.2f} "
                    f"gpu_peak_reserved_mb={self.gpu_peak_memory_reserved_mb:.2f} "
                    f"cpu_rss_mb={self.cpu_rss_mb:.2f}")
        elif self.device_type == "mps":
            return (f"device={self.device_type} "
                    f"gpu_peak_stats=not_available "
                    f"cpu_rss_mb={self.cpu_rss_mb:.2f}")
        else:
            return f"device={self.device_type} cpu_rss_mb={self.cpu_rss_mb:.2f}"


@dataclass
class DatasetStats:
    """Container for dataset statistics."""
    dataset_name: str
    num_series: int
    num_groups: int  # Estimated as batch_size when cross_learning=True
    avg_group_size: float
    seq_len: int  # Context length
    d_model: int
    prediction_length: int
    
    def to_log_string(self) -> str:
        """Format as single-line key=value string for easy parsing."""
        return (f"dataset={self.dataset_name} "
                f"num_series={self.num_series} "
                f"num_groups={self.num_groups} "
                f"avg_group_size={self.avg_group_size:.2f} "
                f"seq_len={self.seq_len} "
                f"d_model={self.d_model} "
                f"prediction_length={self.prediction_length}")


@dataclass 
class ComplexityStats:
    """Container for complexity estimates."""
    cross_group_attention_ops: int  # O(G² * D)
    broadcast_fusion_ops: int  # O(B * T * D)
    
    def to_log_string(self) -> str:
        """Format as single-line key=value string."""
        return (f"cross_group_attn_ops={self.cross_group_attention_ops} "
                f"broadcast_fusion_ops={self.broadcast_fusion_ops}")


def get_memory_stats(device: str) -> MemoryStats:
    """
    Get current memory statistics for the given device.
    
    Supports:
    - CUDA: Full GPU memory stats via torch.cuda
    - MPS: CPU RSS only (GPU stats not available)
    - CPU: CPU RSS via psutil
    """
    stats = MemoryStats(device_type="cpu")
    
    # Get CPU RSS via psutil
    try:
        import psutil
        process = psutil.Process(os.getpid())
        stats.cpu_rss_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        stats.cpu_rss_mb = -1.0  # psutil not available
    
    if device.startswith("cuda"):
        stats.device_type = "cuda"
        stats.gpu_memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        stats.gpu_memory_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        stats.gpu_peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        stats.gpu_peak_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    elif device == "mps":
        stats.device_type = "mps"
        # MPS doesn't expose memory stats like CUDA
        # Best effort: just use CPU RSS
    
    return stats


def reset_memory_stats(device: str):
    """Reset peak memory statistics (CUDA only)."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def estimate_complexity(
    batch_size: int,
    num_groups: int,
    seq_len: int,
    d_model: int,
) -> ComplexityStats:
    """
    Estimate computational complexity for cross-group attention.
    
    - Cross-group attention: O(G² * D) where G = num_groups, D = d_model
    - Broadcast fusion: O(B * T * D) where B = batch, T = seq_len
    """
    cross_group_attn_ops = num_groups * num_groups * d_model
    broadcast_fusion_ops = batch_size * seq_len * d_model
    
    return ComplexityStats(
        cross_group_attention_ops=cross_group_attn_ops,
        broadcast_fusion_ops=broadcast_fusion_ops,
    )


def log_dataset_evaluation_start(
    dataset_name: str,
    num_series: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    prediction_length: int,
    device: str,
):
    """Log statistics at the start of dataset evaluation."""
    # For cross_learning, each batch item is its own group
    # So num_groups ≈ batch_size, avg_group_size ≈ 1
    num_groups = min(batch_size, num_series)
    avg_group_size = num_series / num_groups if num_groups > 0 else 0
    
    dataset_stats = DatasetStats(
        dataset_name=dataset_name,
        num_series=num_series,
        num_groups=num_groups,
        avg_group_size=avg_group_size,
        seq_len=seq_len,
        d_model=d_model,
        prediction_length=prediction_length,
    )
    
    complexity_stats = estimate_complexity(batch_size, num_groups, seq_len, d_model)
    
    reset_memory_stats(device)
    
    logger.info(f"[STATS_START] {dataset_stats.to_log_string()}")
    logger.info(f"[COMPLEXITY] {complexity_stats.to_log_string()}")


def log_dataset_evaluation_end(
    dataset_name: str,
    device: str,
    wall_time_seconds: float,
    mase: float,
    wql: float,
):
    """Log statistics at the end of dataset evaluation."""
    memory_stats = get_memory_stats(device)
    
    logger.info(f"[STATS_END] dataset={dataset_name} "
                f"wall_time_sec={wall_time_seconds:.2f} "
                f"mase={mase:.6f} wql={wql:.6f} "
                f"{memory_stats.to_log_string()}")


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


def enable_cross_group_attention(
    pipeline: Chronos2Pipeline,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    use_sparse_routing: bool = False,
    routing_temperature: float = 1.0,
    always_include_self: bool = True,
) -> Chronos2Pipeline:
    """
    Enable selective cross-group attention in an existing Chronos2 pipeline.
    
    This modifies the model's config and reinitializes the encoder blocks
    to include the CrossGroupAttention layers with selective sharing.
    
    Args:
        pipeline: Base Chronos2Pipeline to modify
        top_k: If set, attend only to top-k most similar groups (+ self)
        similarity_threshold: If set, mask attention below this cosine similarity
        use_sparse_routing: If True, use learned sparse routing network
        routing_temperature: Temperature for sparse routing softmax
        always_include_self: Always include self-attention in top-k/threshold
    
    Returns:
        New Chronos2Pipeline with cross-group attention enabled
    """
    # Log configuration
    config_str = (f"top_k={top_k} similarity_threshold={similarity_threshold} "
                  f"sparse_routing={use_sparse_routing} temp={routing_temperature} "
                  f"include_self={always_include_self}")
    logger.info(f"Enabling Selective Cross-Group Attention: {config_str}")
    
    # Get the current config
    config = deepcopy(pipeline.model.config)
    
    # Enable cross-group attention with selective settings
    config.use_cross_group_attention = True
    config.cross_group_top_k = top_k
    config.cross_group_similarity_threshold = similarity_threshold
    config.cross_group_use_sparse_routing = use_sparse_routing
    config.cross_group_routing_temperature = routing_temperature
    config.cross_group_always_include_self = always_include_self
    
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
    
    logger.info("Cross-Group Attention layers initialized successfully!")
    
    return Chronos2Pipeline(model=new_model)


def evaluate_model(
    pipeline: Chronos2Pipeline,
    backtest_configs: list,
    batch_size: int,
    device: str,
    enable_cross_learning: bool = False,
    model_name: str = "baseline",
    enable_instrumentation: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a model on all datasets in the config with memory/complexity instrumentation.
    
    Args:
        pipeline: Chronos2Pipeline to evaluate
        backtest_configs: List of dataset configs
        batch_size: Batch size for inference
        device: Device string ("cuda", "cpu", "mps")
        enable_cross_learning: Whether to enable cross-learning
        model_name: Name for the model in results
        enable_instrumentation: Whether to log memory/complexity stats
    
    Returns:
        DataFrame with evaluation results
    """
    result_rows = []
    
    # Get model dimensions for complexity estimation
    d_model = pipeline.model.config.d_model
    # Estimate context length from chronos config
    chronos_config = getattr(pipeline.model.config, 'chronos_config', {})
    if isinstance(chronos_config, dict):
        context_length = chronos_config.get('context_length', 512)
    else:
        context_length = getattr(chronos_config, 'context_length', 512)
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(backtest_config=config)
        num_series = len(test_data.input)

        # Log stats at start of evaluation
        if enable_instrumentation:
            log_dataset_evaluation_start(
                dataset_name=dataset_name,
                num_series=num_series,
                batch_size=batch_size,
                seq_len=context_length,
                d_model=d_model,
                prediction_length=prediction_length,
                device=device,
            )

        logger.info(f"Generating forecasts for {dataset_name} ({num_series} time series)")
        
        # Track wall-clock time
        start_time = time.time()
        
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
        
        wall_time = time.time() - start_time
        
        # Extract metrics
        mase_value = metrics[0].get("MASE[0.5]", 0.0)
        wql_value = metrics[0].get("mean_weighted_sum_quantile_loss", 0.0)
        
        # Log stats at end of evaluation
        if enable_instrumentation:
            log_dataset_evaluation_end(
                dataset_name=dataset_name,
                device=device,
                wall_time_seconds=wall_time,
                mase=mase_value,
                wql=wql_value,
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
    # Selective cross-group attention options
    cross_group_top_k: int = typer.Option(
        None,
        help="Top-k groups to attend to (None = all groups)"
    ),
    cross_group_sim_threshold: float = typer.Option(
        None,
        help="Cosine similarity threshold for attention masking (None = no threshold)"
    ),
    cross_group_sparse_routing: bool = typer.Option(
        False,
        help="Use learned sparse routing"
    ),
    cross_group_routing_temp: float = typer.Option(
        1.0,
        help="Temperature for sparse routing"
    ),
):
    """
    Evaluate Selective Cross-Group Attention mechanism against baseline Chronos-2.
    
    This script compares:
    1. Baseline Chronos-2 (no cross-group attention)
    2. Chronos-2 with cross_learning=True (simple approach)
    3. Chronos-2 with Selective Cross-Group Attention (our novel approach)
    
    Selective mechanisms (use ONE):
    - --cross-group-top-k N: Attend only to top-N most similar groups
    - --cross-group-sim-threshold T: Mask attention below cosine similarity T
    - --cross-group-sparse-routing: Use learned sparse routing network
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
        device=device,
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
            device=device,
            enable_cross_learning=True,
            model_name="chronos2_cross_learning"
        )
        all_results.append(cross_learning_results)
        logger.info(f"\nCross-Learning Results:\n{cross_learning_results.to_string()}")
    
    # 3. Evaluate with Selective Cross-Group Attention
    logger.info("=" * 60)
    logger.info("Evaluating with SELECTIVE CROSS-GROUP ATTENTION")
    logger.info("=" * 60)
    
    # Build model name based on selective mechanism
    model_name = "chronos2_cross_group"
    if cross_group_top_k is not None:
        model_name += f"_topk{cross_group_top_k}"
    elif cross_group_sim_threshold is not None:
        model_name += f"_simthresh{cross_group_sim_threshold}"
    elif cross_group_sparse_routing:
        model_name += f"_sparse_temp{cross_group_routing_temp}"
    else:
        model_name += "_full"
    
    cross_group_pipeline = enable_cross_group_attention(
        baseline_pipeline,
        top_k=cross_group_top_k,
        similarity_threshold=cross_group_sim_threshold,
        use_sparse_routing=cross_group_sparse_routing,
        routing_temperature=cross_group_routing_temp,
    )
    cross_group_results = evaluate_model(
        cross_group_pipeline,
        backtest_configs,
        batch_size,
        device=device,
        enable_cross_learning=True,  # Use cross_learning to ensure groups are defined
        model_name=model_name
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
    if "chronos2_baseline" in pivot_mase.columns and model_name in pivot_mase.columns:
        mase_improvement = (pivot_mase["chronos2_baseline"] - pivot_mase[model_name]) / pivot_mase["chronos2_baseline"] * 100
        wql_improvement = (pivot_wql["chronos2_baseline"] - pivot_wql[model_name]) / pivot_wql["chronos2_baseline"] * 100
        
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
def ablation(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "cross_group_test.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/cross_group_ablation.csv"),
        help="Path to save the ablation results CSV"
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
):
    """
    Run ablation study over selective cross-group attention configurations.
    
    Sweeps over:
    - Top-k values: [1, 2, 3, 5, None (full)]
    - Similarity thresholds: [0.3, 0.5, 0.7, 0.9]
    
    Outputs a comprehensive comparison CSV for analysis.
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
    
    # 1. Baseline (no cross-learning)
    logger.info("=" * 60)
    logger.info("ABLATION: Baseline (no cross-learning)")
    logger.info("=" * 60)
    baseline_results = evaluate_model(
        baseline_pipeline, backtest_configs, batch_size, device=device,
        enable_cross_learning=False, model_name="baseline"
    )
    all_results.append(baseline_results)
    
    # 2. Cross-learning only
    logger.info("=" * 60)
    logger.info("ABLATION: Cross-learning only")
    logger.info("=" * 60)
    cross_learning_results = evaluate_model(
        baseline_pipeline, backtest_configs, batch_size, device=device,
        enable_cross_learning=True, model_name="cross_learning"
    )
    all_results.append(cross_learning_results)
    
    # 3. Full cross-group attention (no selective mechanism)
    logger.info("=" * 60)
    logger.info("ABLATION: Full cross-group attention")
    logger.info("=" * 60)
    full_cga_pipeline = enable_cross_group_attention(baseline_pipeline)
    full_cga_results = evaluate_model(
        full_cga_pipeline, backtest_configs, batch_size, device=device,
        enable_cross_learning=True, model_name="cga_full"
    )
    all_results.append(full_cga_results)
    
    # 4. Top-k ablation
    top_k_values = [1, 2, 3, 5]
    for k in top_k_values:
        logger.info("=" * 60)
        logger.info(f"ABLATION: Top-k={k}")
        logger.info("=" * 60)
        topk_pipeline = enable_cross_group_attention(baseline_pipeline, top_k=k)
        topk_results = evaluate_model(
            topk_pipeline, backtest_configs, batch_size, device=device,
            enable_cross_learning=True, model_name=f"cga_topk{k}"
        )
        all_results.append(topk_results)
    
    # 5. Similarity threshold ablation
    threshold_values = [0.3, 0.5, 0.7, 0.9]
    for thresh in threshold_values:
        logger.info("=" * 60)
        logger.info(f"ABLATION: Similarity threshold={thresh}")
        logger.info("=" * 60)
        thresh_pipeline = enable_cross_group_attention(baseline_pipeline, similarity_threshold=thresh)
        thresh_results = evaluate_model(
            thresh_pipeline, backtest_configs, batch_size, device=device,
            enable_cross_learning=True, model_name=f"cga_thresh{thresh}"
        )
        all_results.append(thresh_results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create summary
    logger.info("=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    
    pivot_mase = combined_results.pivot(index="dataset", columns="model", values="MASE")
    pivot_wql = combined_results.pivot(index="dataset", columns="model", values="WQL")
    
    logger.info("\nMASE by model:")
    logger.info(pivot_mase.to_string())
    
    logger.info("\nWQL by model:")
    logger.info(pivot_wql.to_string())
    
    # Compute average improvement over baseline for each model
    logger.info("\n" + "=" * 60)
    logger.info("AVERAGE IMPROVEMENT OVER BASELINE")
    logger.info("=" * 60)
    
    for model in pivot_mase.columns:
        if model != "baseline":
            mase_imp = ((pivot_mase["baseline"] - pivot_mase[model]) / pivot_mase["baseline"] * 100).mean()
            wql_imp = ((pivot_wql["baseline"] - pivot_wql[model]) / pivot_wql["baseline"] * 100).mean()
            logger.info(f"{model}: MASE={mase_imp:+.2f}%, WQL={wql_imp:+.2f}%")
    
    # Save results
    combined_results.to_csv(output_path, index=False)
    logger.info(f"\nAblation results saved to {output_path}")
    
    return combined_results


@app.command()
def quick_test(
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    ),
):
    """
    Quick smoke test to verify Selective Cross-Group Attention implementation works.
    Tests: full attention, top-k, similarity threshold, and sparse routing.
    """
    logger.info("Running quick smoke test for Selective Cross-Group Attention...")
    
    # Common test parameters
    batch_size = 6
    context_length = 64
    base_chronos_config = {
        "context_length": 64,
        "output_patch_size": 8,
        "input_patch_size": 8,
        "input_patch_stride": 8,
        "quantiles": [0.1, 0.5, 0.9],
        "use_reg_token": True,
    }
    
    # Test configurations: (name, config_overrides)
    test_configs = [
        ("Full attention (no selectivity)", {}),
        ("Top-k=2", {"cross_group_top_k": 2}),
        ("Similarity threshold=0.5", {"cross_group_similarity_threshold": 0.5}),
        ("Sparse routing (temp=0.5)", {"cross_group_use_sparse_routing": True, "cross_group_routing_temperature": 0.5}),
    ]
    
    for test_name, config_overrides in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_name}")
        logger.info(f"{'='*50}")
        
        config = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,
            use_cross_group_attention=True,
            chronos_config=base_chronos_config,
            **config_overrides,
        )
        
        model = Chronos2Model(config).to(device)
        model.eval()
        
        # Create dummy inputs with 3 groups
        context = torch.randn(batch_size, context_length, device=device)
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
        
        logger.info(f"  batch_size={batch_size}, num_groups=3, group_ids={group_ids.tolist()}")
        
        try:
            with torch.no_grad():
                output = model(
                    context=context,
                    group_ids=group_ids,
                    num_output_patches=1,
                    output_attentions=True,
                )
            
            logger.info(f"  ✅ Output shape: {output.quantile_preds.shape}")
            
            if output.enc_cross_group_attn_weights is not None:
                for i, weights in enumerate(output.enc_cross_group_attn_weights):
                    logger.info(f"    Layer {i} attn weights shape: {weights.shape}")
                    # Log attention distribution for first layer
                    if i == 0:
                        logger.info(f"    Layer {i} attn weights:\n{weights.detach().cpu().numpy()}")
            else:
                logger.info("  ℹ️  No cross-group attention weights (expected for single group)")
                
        except Exception as e:
            logger.error(f"  ❌ FAILED: {e}")
            raise
    
    # Test edge case: single group (should skip cross-group attention)
    logger.info(f"\n{'='*50}")
    logger.info("Testing: Single group edge case")
    logger.info(f"{'='*50}")
    
    config = Chronos2CoreConfig(
        d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
        dropout_rate=0.1, use_cross_group_attention=True,
        chronos_config=base_chronos_config,
    )
    model = Chronos2Model(config).to(device)
    model.eval()
    
    single_group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    context = torch.randn(batch_size, context_length, device=device)
    
    with torch.no_grad():
        output = model(context=context, group_ids=single_group_ids, num_output_patches=1)
    
    logger.info(f"  ✅ Single group handled correctly, output shape: {output.quantile_preds.shape}")
    
    logger.info("\n" + "="*50)
    logger.info("✅ ALL QUICK TESTS PASSED!")
    logger.info("="*50)


if __name__ == "__main__":
    app()
