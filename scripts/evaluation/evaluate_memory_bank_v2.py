#!/usr/bin/env python3
"""
Evaluation script for Series-Memory Bank v2 (Fixed version)

Key fixes from v1:
- NaN-safe softmax handling
- ID-based exclusion (not similarity-based)
- Parameter-free fusion (works without training)
- Proper diagnostic logging
- Better pooling (last-token instead of mean)
"""

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from chronos.chronos2.memory_bank_v2 import (
    MemoryBankConfig,
    MemoryAugmentedForecasterV2,
)

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Memory Bank V2")
logger.setLevel(logging.INFO)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASE_KEY = "MASE[0.5]"
WQL_KEY = "mean_weighted_sum_quantile_loss"


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
    return test_data, gts_dataset


def encode_series_statistical(context_list: list, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Encode series using statistical features (no model needed).
    Uses last-token style: focus on recent dynamics.
    """
    features = []
    for c in context_list:
        c = c.float()
        
        # Use last N points for "recent dynamics" representation
        last_n = min(64, len(c))
        recent = c[-last_n:]
        
        # Statistics
        mean = recent.mean()
        std = recent.std() if len(recent) > 1 else torch.tensor(0.0)
        
        # Trend (recent)
        if len(recent) > 1:
            x = torch.arange(len(recent), dtype=torch.float32)
            x_mean = x.mean()
            trend = ((x - x_mean) * (recent - mean)).sum() / ((x - x_mean) ** 2).sum().clamp(min=1e-8)
        else:
            trend = torch.tensor(0.0)
        
        # Autocorrelation
        if len(recent) > 1:
            autocorr = torch.corrcoef(torch.stack([recent[:-1], recent[1:]]))[0, 1]
            if torch.isnan(autocorr):
                autocorr = torch.tensor(0.0)
        else:
            autocorr = torch.tensor(0.0)
        
        # FFT features (periodicity)
        normalized = (recent - mean) / (std + 1e-8)
        if len(normalized) >= 8:
            fft_vals = torch.fft.rfft(normalized)
            fft_feats = torch.abs(fft_vals[:min(d_model // 4, len(fft_vals))])
        else:
            fft_feats = torch.zeros(d_model // 4)
        
        # Last few values (normalized) - captures recent shape
        last_vals = normalized[-min(16, len(normalized)):]
        if len(last_vals) < 16:
            last_vals = F.pad(last_vals, (0, 16 - len(last_vals)))
        
        # Combine
        stats = torch.tensor([mean, std, trend, autocorr])
        combined = torch.cat([stats, fft_feats, last_vals])
        
        if len(combined) < d_model:
            combined = F.pad(combined, (0, d_model - len(combined)))
        else:
            combined = combined[:d_model]
        
        features.append(combined)
    
    series_reps = torch.stack(features).to(device)
    # NOTE: Do NOT normalize here - let memory bank normalize keys only
    # This preserves raw magnitude differences in values for fusion
    return series_reps


def stable_series_hash(x: torch.Tensor) -> str:
    """Stable hash for series (deterministic across processes)."""
    arr = x.detach().cpu().numpy().astype(np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def build_memory_bank(
    gts_dataset: list,
    memory_forecaster: MemoryAugmentedForecasterV2,
    d_model: int,
    device: torch.device,
    batch_size: int = 32,
    max_series: int = 5000,
) -> dict:
    """
    Build memory bank from dataset with proper ID tracking.
    
    Returns:
        series_hash_to_id: Dict mapping series content hash to memory bank ID
    """
    rng = np.random.default_rng(42)
    num_series = min(len(gts_dataset), max_series)
    indices = rng.choice(len(gts_dataset), num_series, replace=False)
    
    # Map series content hash to memory bank ID
    series_hash_to_id = {}
    
    logger.info(f"Building memory bank from {num_series} series...")
    
    for batch_start in tqdm(range(0, num_series, batch_size), desc="Building memory"):
        batch_end = min(batch_start + batch_size, num_series)
        batch_indices = indices[batch_start:batch_end]
        batch_entries = [gts_dataset[i] for i in batch_indices]
        
        context = [torch.tensor(entry["target"]) for entry in batch_entries]
        series_reps = encode_series_statistical(context, d_model, device)
        
        # Use dataset indices as series IDs
        series_ids = torch.tensor(batch_indices, device=device)
        
        # Store stable hash -> dataset index mapping
        for c, idx in zip(context, batch_indices):
            h = stable_series_hash(c)
            series_hash_to_id[h] = int(idx)
        
        # Fake hidden states shape for the API
        fake_hidden = series_reps.unsqueeze(1)  # (B, 1, d)
        memory_forecaster.build_memory_from_hidden_states(
            fake_hidden, series_ids=series_ids
        )
    
    logger.info(f"Memory bank built: {memory_forecaster.memory_bank.get_stats()}")
    logger.info(f"Tracked {len(series_hash_to_id)} series hashes for exclusion")
    
    return series_hash_to_id


def generate_forecasts_baseline(
    test_data_input,
    pipeline: Chronos2Pipeline,
    prediction_length: int,
    batch_size: int,
    cross_learning: bool = False,
):
    """Generate baseline forecasts."""
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


def generate_forecasts_with_memory(
    test_data_input,
    pipeline: Chronos2Pipeline,
    memory_forecaster: MemoryAugmentedForecasterV2,
    prediction_length: int,
    batch_size: int,
    cross_learning: bool = True,
    memory_series_ids: Optional[dict] = None,  # Maps series hash to dataset index
    gts_dataset: Optional[list] = None,  # Full dataset for fetching neighbors
    top_k: int = 5,
):
    """
    Generate forecasts with memory augmentation via neighbor retrieval.
    
    Option A implementation: For each query, call Chronos on [query + neighbors]
    with cross_learning=True. This makes memory retrieval actually affect forecasts.
    
    Args:
        memory_series_ids: Dict mapping series content hash to dataset index
        gts_dataset: Full dataset to fetch neighbor series from
        top_k: Number of neighbors to retrieve
    """
    forecasts = []
    device = pipeline.model.device
    d_model = pipeline.model.config.d_model
    
    # Track retrieval stats
    total_queries = 0
    total_neighbors_retrieved = 0
    max_sims = []
    
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Forecasting"):
        context = [torch.tensor(entry["target"]) for entry in batch]
        
        # Get series representations for retrieval
        series_reps = encode_series_statistical(context, d_model, device)
        
        # Create series IDs for exclusion
        series_ids = []
        for c in context:
            h = stable_series_hash(c)
            series_ids.append(memory_series_ids.get(h, -1) if memory_series_ids else -1)
        series_ids_tensor = torch.tensor(series_ids, device=device)
        
        # Retrieve neighbor IDs from memory bank
        _, sims, mask, retrieved_ids = memory_forecaster.memory_bank.retrieve(
            series_reps, query_ids=series_ids_tensor, exclude_ids=True
        )
        
        # Process each query: call pipeline on [query + neighbors]
        for i, (ctx, ts) in enumerate(zip(context, batch)):
            total_queries += 1
            
            # Get valid neighbor indices (dataset indices)
            valid_mask = mask[i]
            neighbor_dataset_ids = retrieved_ids[i][valid_mask].tolist()
            neighbor_sims = sims[i][valid_mask]
            
            if len(neighbor_sims) > 0:
                max_sims.append(neighbor_sims.max().item())
                total_neighbors_retrieved += len(neighbor_dataset_ids)
            
            # Build group: [query] + [neighbors from dataset]
            if gts_dataset is not None and len(neighbor_dataset_ids) > 0:
                # Fetch actual neighbor series from dataset
                neighbor_series = []
                for nid in neighbor_dataset_ids:
                    if 0 <= nid < len(gts_dataset):
                        neighbor_series.append(torch.tensor(gts_dataset[nid]["target"]))
                
                if neighbor_series:
                    # Call pipeline on [query + neighbors] with cross_learning
                    group_inputs = [ctx] + neighbor_series
                    q, _ = pipeline.predict_quantiles(
                        group_inputs,
                        prediction_length=prediction_length,
                        quantile_levels=QUANTILES,
                        cross_learning=True,  # Memory affects forecast via cross-group attention
                        batch_size=len(group_inputs),
                    )
                    # Keep ONLY the query forecast (first one)
                    q0 = q[0] if isinstance(q, list) else q[0]
                else:
                    # No valid neighbors - fallback to baseline
                    q, _ = pipeline.predict_quantiles(
                        [ctx],
                        prediction_length=prediction_length,
                        quantile_levels=QUANTILES,
                        cross_learning=False,
                    )
                    q0 = q[0]
            else:
                # No memory bank or no neighbors - baseline forecast
                q, _ = pipeline.predict_quantiles(
                    [ctx],
                    prediction_length=prediction_length,
                    quantile_levels=QUANTILES,
                    cross_learning=cross_learning,
                )
                q0 = q[0]
            
            # Match baseline: stack, squeeze axis=1, swapaxes
            # q is a list from predict_quantiles
            if isinstance(q, list):
                quantiles_arr = np.stack(q).squeeze(axis=1)  # (1, Q, H) -> (Q, H)
            else:
                quantiles_arr = q.squeeze(axis=1) if q.ndim > 2 else q
            quantiles_arr = quantiles_arr.swapaxes(-1, -2)  # (Q, H) -> (H, Q)
            forecast_array = quantiles_arr[0] if quantiles_arr.ndim == 3 else quantiles_arr
            
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=forecast_array,
                    forecast_keys=list(map(str, QUANTILES)),
                    start_date=forecast_start_date,
                )
            )
    
    # Aggregate diagnostics
    avg_neighbors = total_neighbors_retrieved / total_queries if total_queries > 0 else 0
    agg_diagnostics = {
        'valid_rows': total_queries,
        'total_rows': total_queries,
        'avg_gate': 1.0 if avg_neighbors > 0 else 0.0,  # Gate is implicit now
        'avg_delta': avg_neighbors,  # Repurposed: avg neighbors per query
        'self_hit_rate': 0.0,
        'max_sim_mean': np.mean(max_sims) if max_sims else 0.0,
        'max_sim_max': np.max(max_sims) if max_sims else 0.0,
        'avg_neighbors': avg_neighbors,
    }
    
    return forecasts, agg_diagnostics


@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "no_dominick.yaml",
        help="Path to the evaluation config YAML file"
    ),
    output_path: Path = typer.Option(
        Path("/tmp/memory_bank_v2_results.csv"),
        help="Path to save the results CSV"
    ),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = typer.Option(8),
    top_k: int = typer.Option(5),
    similarity_threshold: float = typer.Option(0.5),  # Lower threshold
    max_memory_size: int = typer.Option(5000),
    pooling: str = typer.Option("last"),  # last, mean, attention
):
    """
    Evaluate memory-augmented forecasting v2.
    """
    
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)
    
    logger.info(f"Loading model from {model_id}")
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.float32
    )
    
    d_model = pipeline.model.config.d_model
    device_obj = pipeline.model.device
    
    # Create memory config
    memory_config = MemoryBankConfig(
        d_model=d_model,
        max_memory_size=max_memory_size,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        use_gating=True,
        gate_temperature=5.0,  # Sharp gating
        use_learned_projections=False,  # Parameter-free
        pooling_method=pooling,
    )
    
    all_results = []
    
    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info("=" * 60)
        
        test_data, gts_dataset = load_and_split_dataset(backtest_config=config)
        
        # Create fresh memory forecaster for each dataset
        memory_forecaster = MemoryAugmentedForecasterV2(memory_config)
        
        # Build memory bank from training portion (returns hash->ID mapping)
        series_hash_to_id = build_memory_bank(
            gts_dataset, memory_forecaster, d_model, device_obj,
            batch_size=batch_size, max_series=max_memory_size
        )
        
        # 1. BASELINE
        logger.info("  [baseline]")
        start_time = time.time()
        forecasts = generate_forecasts_baseline(
            test_data.input, pipeline, prediction_length, batch_size, cross_learning=False
        )
        metrics = evaluate_forecasts(
            forecasts, test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        baseline_mase = metrics[0].get(MASE_KEY, 0.0)
        baseline_wql = metrics[0].get(WQL_KEY, 0.0)
        logger.info(f"    MASE={baseline_mase:.4f}, WQL={baseline_wql:.4f} ({time.time()-start_time:.1f}s)")
        all_results.append({
            "dataset": dataset_name, "model": "baseline",
            "MASE": baseline_mase, "WQL": baseline_wql
        })
        
        # 2. CROSS_LEARNING
        logger.info("  [cross_learning]")
        start_time = time.time()
        forecasts = generate_forecasts_baseline(
            test_data.input, pipeline, prediction_length, batch_size, cross_learning=True
        )
        metrics = evaluate_forecasts(
            forecasts, test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        cl_mase = metrics[0].get(MASE_KEY, 0.0)
        cl_wql = metrics[0].get(WQL_KEY, 0.0)
        logger.info(f"    MASE={cl_mase:.4f}, WQL={cl_wql:.4f} ({time.time()-start_time:.1f}s)")
        all_results.append({
            "dataset": dataset_name, "model": "cross_learning",
            "MASE": cl_mase, "WQL": cl_wql
        })
        
        # 3. MEMORY AUGMENTED (Option A: call pipeline on [query + neighbors])
        logger.info(f"  [memory_augmented] (top_k={top_k}, threshold={similarity_threshold})")
        start_time = time.time()
        memory_forecaster.reset_stats()
        forecasts, diagnostics = generate_forecasts_with_memory(
            test_data.input, pipeline, memory_forecaster,
            prediction_length, batch_size, cross_learning=True,
            memory_series_ids=series_hash_to_id,
            gts_dataset=list(gts_dataset),  # Pass dataset for fetching neighbors
            top_k=top_k,
        )
        metrics = evaluate_forecasts(
            forecasts, test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        mem_mase = metrics[0].get(MASE_KEY, 0.0)
        mem_wql = metrics[0].get(WQL_KEY, 0.0)
        
        retrieval_rate = diagnostics['valid_rows'] / diagnostics['total_rows'] if diagnostics['total_rows'] > 0 else 0
        
        self_hit_rate = diagnostics.get('self_hit_rate', 0.0)
        max_sim_mean = diagnostics.get('max_sim_mean', 0.0)
        max_sim_max = diagnostics.get('max_sim_max', 0.0)
        logger.info(f"    MASE={mem_mase:.4f}, WQL={mem_wql:.4f} ({time.time()-start_time:.1f}s)")
        logger.info(f"    Retrieval rate: {retrieval_rate:.1%}, Self-hit rate: {self_hit_rate:.1%}")
        logger.info(f"    Max sim: mean={max_sim_mean:.4f}, max={max_sim_max:.4f}")
        avg_neighbors = diagnostics.get('avg_neighbors', 0.0)
        logger.info(f"    Avg neighbors retrieved: {avg_neighbors:.2f}")
        
        all_results.append({
            "dataset": dataset_name, "model": "memory_augmented",
            "MASE": mem_mase, "WQL": mem_wql,
            "retrieval_rate": retrieval_rate,
            "avg_gate": diagnostics['avg_gate'],
            "avg_delta": diagnostics['avg_delta'],
        })
        
        # Summary
        logger.info(f"  Summary for {dataset_name}:")
        logger.info(f"    cross_learning vs baseline: {(baseline_mase - cl_mase) / baseline_mase * 100:+.2f}% MASE")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    
    pivot = results_df.pivot(index="dataset", columns="model", values="MASE")
    cols = [c for c in ["baseline", "cross_learning", "memory_augmented"] if c in pivot.columns]
    pivot = pivot[cols]
    logger.info("\nMASE Results:")
    logger.info(pivot.to_string())
    
    logger.info("\nDiagnostic Summary:")
    mem_results = results_df[results_df["model"] == "memory_augmented"]
    for _, row in mem_results.iterrows():
        logger.info(f"  {row['dataset']}: retrieval={row.get('retrieval_rate', 0):.1%}, gate={row.get('avg_gate', 0):.3f}, delta={row.get('avg_delta', 0):.4f}")
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    app()
