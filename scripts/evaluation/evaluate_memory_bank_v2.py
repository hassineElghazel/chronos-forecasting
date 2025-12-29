#!/usr/bin/env python3
"""
Evaluation script for Series-Memory Bank v2 (Option A, corrected + improved gating)

Key features:
1) Leakage-safe memory: build memory ONLY from the TRAIN split (pre-offset).
2) Stable IDs: add item_id per univariate series and use it everywhere (retrieval + exclusion).
3) Neighbor slicing: when using neighbors, slice neighbor context to the SAME length as the query context.
4) Meaningful gating: threshold + "peakedness" (top1 - top2 >= gap_threshold).
5) Decouple retrieval from gating:
   - retrieve top N (retrieve_k) with a low retrieval threshold
   - use at most M (use_k) neighbors only when gate passes
6) Optional safety: disable memory augmentation when train bank is too small (min_train_series_for_memory).

IMPORTANT PERF FIX (your 4s/it issue):
- Previously, Option A was calling pipeline.predict_quantiles() PER SERIES for all non-gated queries.
  That kills GPU utilization and adds huge Python/dispatch overhead.
- Now: within each batch, we run ONE batched baseline call for all non-gated queries,
  and only run per-query calls for the gated ones (typically a small fraction).
"""

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from chronos import BaseChronosPipeline, Chronos2Pipeline  # noqa: F401
from chronos.chronos2.memory_bank_v2 import MemoryBankConfig, MemoryAugmentedForecasterV2

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Memory Bank V2 (Option A)")
logger.setLevel(logging.INFO)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASE_KEY = "MASE[0.5]"
WQL_KEY = "mean_weighted_sum_quantile_loss"


# -----------------------------
# Dataset conversion + split
# -----------------------------
def to_gluonts_univariate(hf_dataset: datasets.Dataset) -> List[dict]:
    """
    Convert multivariate HF dataset -> list of univariate GluonTS entries.
    IMPORTANT: assigns a stable item_id for each produced univariate series.
    """
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    if "timestamp" in series_fields:
        series_fields.remove("timestamp")

    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    item_id = 0
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(hf_entry["timestamp"][0], freq=dataset_freq),
                    "target": hf_entry[field],
                    "item_id": int(item_id),
                }
            )
            item_id += 1
    return gts_dataset


def load_and_split_dataset(backtest_config: dict) -> Tuple:
    """
    Returns:
      test_data (GluonTS Instances)
      train_list (list of train entries, leakage-safe)
    """
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False
    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(ds)

    train_data, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    train_list = list(train_data)
    train_list.sort(key=lambda x: int(x.get("item_id", 0)))

    return test_data, train_list


# -----------------------------
# Series embedding for retrieval
# -----------------------------
def stable_series_hash(x: torch.Tensor) -> str:
    """Stable hash for a series tensor (deterministic across processes)."""
    arr = x.detach().cpu().numpy().astype(np.float32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return x.std(unbiased=False).clamp(min=1e-6)


def encode_series_statistical(
    context_list: List[torch.Tensor],
    d_model: int,
    device: torch.device,
    last_n: int = 64,
) -> torch.Tensor:
    """
    Parameter-free representation for cosine retrieval.
    Design goals:
      - focus on recent dynamics (last_n)
      - avoid similarity saturation (block normalization)
      - avoid big zero-padding (fill remaining dims by interpolation)
    """
    feats = []
    for c in context_list:
        c = c.to(device=device).float()
        if c.numel() == 0:
            feats.append(torch.zeros(d_model, device=device))
            continue

        recent = c[-min(last_n, int(c.numel())) :]
        mean = recent.mean()
        std = _safe_std(recent)

        # Trend (slope)
        if recent.numel() > 1:
            x = torch.arange(recent.numel(), device=device, dtype=torch.float32)
            x_mean = x.mean()
            denom = ((x - x_mean) ** 2).sum().clamp(min=1e-8)
            trend = ((x - x_mean) * (recent - mean)).sum() / denom
        else:
            trend = torch.tensor(0.0, device=device)

        # Lag-1 autocorr (safe)
        if recent.numel() > 2:
            a = recent[:-1]
            b = recent[1:]
            a = (a - a.mean()) / _safe_std(a)
            b = (b - b.mean()) / _safe_std(b)
            autocorr = (a * b).mean().clamp(min=-1.0, max=1.0)
        else:
            autocorr = torch.tensor(0.0, device=device)

        # Normalize recent
        normalized = (recent - mean) / (std + 1e-8)

        # FFT magnitude
        fft_budget = max(8, d_model // 4)
        if normalized.numel() >= 8:
            fft_vals = torch.fft.rfft(normalized)
            fft_mag = torch.abs(fft_vals)
            fft_feats = fft_mag[: min(int(fft_budget), int(fft_mag.numel()))]
        else:
            fft_feats = torch.zeros(min(int(fft_budget), d_model), device=device)

        # Last values block
        last_block = min(16, d_model)
        last_vals = normalized[-min(last_block, int(normalized.numel())) :]
        if last_vals.numel() < last_block:
            last_vals = F.pad(last_vals, (0, last_block - int(last_vals.numel())), value=0.0)

        # Stats block
        stats = torch.stack([mean, std, trend, autocorr])

        # Block-normalize
        stats_n = F.normalize(stats, p=2, dim=0) if stats.norm(p=2) > 0 else stats
        fft_n = F.normalize(fft_feats, p=2, dim=0) if fft_feats.norm(p=2) > 0 else fft_feats
        last_nrm = F.normalize(last_vals, p=2, dim=0) if last_vals.norm(p=2) > 0 else last_vals

        # Fill remaining dims by interpolating normalized recent
        used = int(stats_n.numel() + fft_n.numel() + last_nrm.numel())
        remaining = max(0, int(d_model - used))
        if remaining > 0:
            if normalized.numel() == 1:
                fill = normalized.repeat(remaining)
            else:
                fill = F.interpolate(
                    normalized.view(1, 1, -1),
                    size=remaining,
                    mode="linear",
                    align_corners=False,
                ).view(-1)
        else:
            fill = torch.tensor([], device=device)

        vec = torch.cat([stats_n, fft_n, last_nrm, fill], dim=0)
        if vec.numel() < d_model:
            vec = F.pad(vec, (0, d_model - int(vec.numel())), value=0.0)
        elif vec.numel() > d_model:
            vec = vec[:d_model]

        feats.append(vec)

    reps = torch.stack(feats, dim=0)
    reps = F.normalize(reps, p=2, dim=-1)
    return reps


# -----------------------------
# Memory bank build (TRAIN ONLY)
# -----------------------------
def build_memory_bank(
    train_list: List[dict],
    memory_forecaster: MemoryAugmentedForecasterV2,
    d_model: int,
    device: torch.device,
    batch_size: int = 32,
    max_series: int = 5000,
) -> Dict[int, dict]:
    """
    Build memory bank from leakage-safe TRAIN entries.

    Returns:
      train_by_id: mapping item_id -> train entry
    """
    rng = np.random.default_rng(42)

    n = min(len(train_list), int(max_series))
    if n <= 0:
        logger.warning("Train list is empty; memory bank will be empty.")
        return {}

    idxs = rng.choice(len(train_list), n, replace=False)

    train_by_id: Dict[int, dict] = {}
    logger.info(f"Building memory bank from {n} TRAIN series...")

    for start in tqdm(range(0, n, batch_size), desc="Building memory (train)"):
        end = min(start + batch_size, n)
        batch_idxs = idxs[start:end]
        batch_entries = [train_list[int(i)] for i in batch_idxs]

        context = [torch.tensor(e["target"]) for e in batch_entries]
        reps = encode_series_statistical(context, d_model, device)

        ids = [int(e.get("item_id", -1)) for e in batch_entries]
        series_ids = torch.tensor(ids, device=device, dtype=torch.long)

        fake_hidden = reps.unsqueeze(1)  # (B,1,d)
        memory_forecaster.build_memory_from_hidden_states(fake_hidden, series_ids=series_ids)

        for e in batch_entries:
            iid = int(e.get("item_id", -1))
            if iid != -1:
                train_by_id[iid] = e

    logger.info(f"Memory bank built: {memory_forecaster.memory_bank.get_stats()}")
    logger.info(f"Train index size: {len(train_by_id)} series (by item_id)")
    return train_by_id


# -----------------------------
# Forecast helpers
# -----------------------------
def _process_single_quantile_output(q_one: np.ndarray) -> np.ndarray:
    """
    q_one is expected like (1, Q, H) or (Q, H).
    Returns (H, Q).
    """
    arr = np.asarray(q_one)
    if arr.ndim == 3:
        arr = arr.squeeze(axis=0)  # (Q,H)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected quantile shape: {arr.shape}")
    return arr.swapaxes(-1, -2)  # (Q,H) -> (H,Q)


def _batch_quantiles_to_bhq(quantiles) -> np.ndarray:
    """
    Convert pipeline.predict_quantiles() output to a numpy array shaped (B, H, Q).
    Accepts either:
      - list of arrays (each like (1,Q,H) or (Q,H))
      - numpy array (B,Q,H) or (B,1,Q,H) etc (handled conservatively)
    """
    if isinstance(quantiles, list):
        # Each element typically (1,Q,H)
        arr = np.stack(quantiles, axis=0)
        # If (B,1,Q,H) -> squeeze the singleton
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr.squeeze(axis=1)  # (B,Q,H)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected stacked quantiles shape: {arr.shape}")
    else:
        arr = np.asarray(quantiles)
        # best-effort squeeze
        while arr.ndim > 3 and arr.shape[1] == 1:
            arr = arr.squeeze(axis=1)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected quantiles ndarray shape: {arr.shape}")

    # (B,Q,H) -> (B,H,Q)
    return arr.swapaxes(-1, -2)


def generate_forecasts_baseline(
    test_data_input_list: List[dict],
    pipeline: Chronos2Pipeline,
    prediction_length: int,
    batch_size: int,
    cross_learning: bool = False,
):
    forecast_outputs = []

    for batch in tqdm(batcher(test_data_input_list, batch_size=batch_size), desc="Forecasting"):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantiles, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=QUANTILES,
            cross_learning=cross_learning,
        )
        bhq = _batch_quantiles_to_bhq(quantiles)
        forecast_outputs.append(bhq)

    forecast_outputs = np.concatenate(forecast_outputs, axis=0)  # (N,H,Q)

    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input_list):
        forecast_start_date = ts["start"] + len(ts["target"])
        forecasts.append(
            QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, QUANTILES)),
                start_date=forecast_start_date,
            )
        )
    return forecasts


def generate_forecasts_with_memory_option_a(
    test_data_input_list: List[dict],
    pipeline: Chronos2Pipeline,
    memory_forecaster: MemoryAugmentedForecasterV2,
    train_by_id: Dict[int, dict],
    prediction_length: int,
    batch_size: int,
    retrieve_k: int,
    use_k: int,
    similarity_threshold: float,
    gap_threshold: float,
    min_train_series_for_memory: int,
):
    """
    Option A:
      - retrieve neighbors (retrieve_k) from memory bank
      - gate using:
          max_sim >= similarity_threshold AND (max_sim - second_sim) >= gap_threshold
      - if gate passes: use up to use_k neighbors, run Chronos on [query + neighbors] with cross_learning=True
      - keep only the query forecast

    PERF:
      - baseline (gate-off) queries are forecasted in one batched call per batch
      - only gate-on queries do per-query calls (small fraction)
    """
    forecasts: List[QuantileForecast] = []
    device = pipeline.model.device
    d_model = pipeline.model.config.d_model

    total_queries = 0
    gated_queries = 0
    retrieved_total = 0
    candidates_total = 0
    max_sims: List[float] = []
    gaps: List[float] = []

    # Dataset-level safety: too-small memory
    if int(min_train_series_for_memory) > 0 and len(train_by_id) < int(min_train_series_for_memory):
        logger.warning(
            f"Memory disabled: train bank too small ({len(train_by_id)} < {min_train_series_for_memory}). "
            "Falling back to baseline for memory(A)."
        )
        forecasts = generate_forecasts_baseline(
            test_data_input_list,
            pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            cross_learning=False,
        )
        diag = {
            "disabled": True,
            "disabled_reason": f"train_bank<{min_train_series_for_memory}",
            "total_rows": len(test_data_input_list),
            "candidate_rate": 0.0,
            "gate_rate": 0.0,
            "avg_neighbors_used": 0.0,
            "max_sim_mean": 0.0,
            "max_sim_p50": 0.0,
            "max_sim_p90": 0.0,
            "max_sim_max": 0.0,
            "gap_mean": 0.0,
            "gap_p50": 0.0,
            "gap_p90": 0.0,
        }
        return forecasts, diag

    for batch in tqdm(batcher(test_data_input_list, batch_size=batch_size), desc="Forecasting (Option A)"):
        # Prepare batch tensors
        context = [torch.tensor(entry["target"]) for entry in batch]
        reps = encode_series_statistical(context, d_model, device)

        # Stable IDs (ideally present through split+instances)
        query_ids = [int(entry.get("item_id", -1)) for entry in batch]
        query_ids_tensor = torch.tensor(query_ids, device=device, dtype=torch.long)

        # Retrieve neighbors for whole batch
        _, sims, mask, retrieved_ids = memory_forecaster.memory_bank.retrieve(
            reps, query_ids=query_ids_tensor, exclude_ids=True
        )

        B = len(batch)
        batch_forecast_arrays: List[np.ndarray] = [None] * B  # each will be (H,Q)

        # Collect non-gated indices for one batched baseline call
        non_gated_indices: List[int] = []
        non_gated_contexts: List[torch.Tensor] = []

        # Store gated groups (index -> group_inputs)
        gated_groups: Dict[int, List[torch.Tensor]] = {}

        for i, (ctx, ts) in enumerate(zip(context, batch)):
            total_queries += 1

            valid_mask = mask[i]
            nb_ids = retrieved_ids[i][valid_mask].tolist()
            nb_sims = sims[i][valid_mask]

            # keep only existing neighbors
            nb_pairs = [
                (int(nid), float(sim))
                for nid, sim in zip(nb_ids, nb_sims)
                if int(nid) in train_by_id
            ]

            if len(nb_pairs) > 0:
                candidates_total += 1

            # defensive sorting and trimming
            nb_pairs.sort(key=lambda x: x[1], reverse=True)
            nb_pairs = nb_pairs[: int(retrieve_k)]

            if len(nb_pairs) == 0:
                # no candidates -> baseline
                non_gated_indices.append(i)
                non_gated_contexts.append(ctx)
                continue

            max_sim = nb_pairs[0][1]
            max_sims.append(max_sim)

            if len(nb_pairs) >= 2:
                gap = max_sim - nb_pairs[1][1]
                gaps.append(gap)
            else:
                gap = -1.0  # no top2 => cannot pass a positive gap_threshold

            use_neighbors = (max_sim >= float(similarity_threshold)) and (gap >= float(gap_threshold))

            if not use_neighbors:
                non_gated_indices.append(i)
                non_gated_contexts.append(ctx)
                continue

            # Gate ON
            gated_queries += 1
            use_pairs = nb_pairs[: int(use_k)]
            retrieved_total += len(use_pairs)

            Lq = int(ctx.numel())
            neighbor_series: List[torch.Tensor] = []
            for nid, _ in use_pairs:
                tgt = np.asarray(train_by_id[nid]["target"], dtype=np.float32)
                tgt = tgt[:Lq]
                neighbor_series.append(torch.tensor(tgt))

            gated_groups[i] = [ctx] + neighbor_series

        # 1) Batched baseline for all non-gated in this batch
        if len(non_gated_contexts) > 0:
            q_list, _ = pipeline.predict_quantiles(
                non_gated_contexts,
                prediction_length=prediction_length,
                quantile_levels=QUANTILES,
                cross_learning=False,
            )
            bhq = _batch_quantiles_to_bhq(q_list)  # (n,H,Q)
            for j, idx in enumerate(non_gated_indices):
                batch_forecast_arrays[idx] = bhq[j]

        # 2) Per-query cross-learning for gated ones (small fraction)
        for idx, group_inputs in gated_groups.items():
            q_list, _ = pipeline.predict_quantiles(
                group_inputs,
                prediction_length=prediction_length,
                quantile_levels=QUANTILES,
                cross_learning=True,
                batch_size=len(group_inputs),
            )
            batch_forecast_arrays[idx] = _process_single_quantile_output(q_list[0])

        # Sanity: ensure all filled
        for i, ts in enumerate(batch):
            if batch_forecast_arrays[i] is None:
                # ultra-defensive fallback (should not happen)
                q_list, _ = pipeline.predict_quantiles(
                    [context[i]],
                    prediction_length=prediction_length,
                    quantile_levels=QUANTILES,
                    cross_learning=False,
                )
                batch_forecast_arrays[i] = _process_single_quantile_output(q_list[0])

            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=batch_forecast_arrays[i],  # (H,Q)
                    forecast_keys=list(map(str, QUANTILES)),
                    start_date=forecast_start_date,
                )
            )

    diag = {
        "disabled": False,
        "disabled_reason": "",
        "total_rows": total_queries,
        "candidate_rate": (candidates_total / total_queries) if total_queries else 0.0,
        "gate_rate": (gated_queries / total_queries) if total_queries else 0.0,
        "avg_neighbors_used": (retrieved_total / gated_queries) if gated_queries else 0.0,
        "max_sim_mean": float(np.mean(max_sims)) if max_sims else 0.0,
        "max_sim_p50": float(np.percentile(max_sims, 50)) if max_sims else 0.0,
        "max_sim_p90": float(np.percentile(max_sims, 90)) if max_sims else 0.0,
        "max_sim_max": float(np.max(max_sims)) if max_sims else 0.0,
        "gap_mean": float(np.mean(gaps)) if gaps else 0.0,
        "gap_p50": float(np.percentile(gaps, 50)) if gaps else 0.0,
        "gap_p90": float(np.percentile(gaps, 90)) if gaps else 0.0,
    }
    return forecasts, diag


# -----------------------------
# CLI entry
# -----------------------------
@app.command()
def evaluate(
    config_path: Path = typer.Option(
        Path(__file__).parent / "configs" / "no_dominick.yaml",
        help="Path to the evaluation config YAML file",
    ),
    output_path: Path = typer.Option(
        Path("/tmp/memory_bank_v2_option_a_results.csv"),
        help="Path to save the results CSV",
    ),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = typer.Option(32),

    # Retrieval vs usage (decoupled)
    retrieve_k: int = typer.Option(5, help="How many neighbors to retrieve (for gating stats like gap)."),
    use_k: int = typer.Option(1, help="How many neighbors to actually use if gate passes."),

    # Gate thresholds
    similarity_threshold: float = typer.Option(0.85, help="Gate: require max cosine sim >= this."),
    gap_threshold: float = typer.Option(0.02, help="Gate: require (top1 - top2) >= this."),

    # Safety
    min_train_series_for_memory: int = typer.Option(
        32, help="Disable memory augmentation if train bank has fewer than this many series (0 disables)."
    ),

    # Memory bank limits
    max_memory_size: int = typer.Option(5000),

    # IMPORTANT: retrieval threshold should be LOW so we can compute gap; gating happens outside.
    retrieval_threshold: float = typer.Option(
        -1.0,
        help="Internal retrieval threshold for the memory bank (keep low, e.g. -1.0 or 0.0).",
    ),

    pooling: str = typer.Option("last"),
):
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    logger.info(f"Loading model from {model_id}")
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.float32,
    )

    d_model = int(pipeline.model.config.d_model)
    device_obj = pipeline.model.device

    # Memory config: set top_k to retrieve_k; keep similarity_threshold LOW here (we gate ourselves).
    memory_config = MemoryBankConfig(
        d_model=d_model,
        max_memory_size=int(max_memory_size),
        top_k=int(retrieve_k),
        similarity_threshold=float(retrieval_threshold),
        use_gating=False,  # we gate explicitly in Option A
        gate_temperature=5.0,
        use_learned_projections=False,
        pooling_method=pooling,
    )

    all_results = []

    for cfg in backtest_configs:
        dataset_name = cfg["name"]
        prediction_length = int(cfg["prediction_length"])

        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info("=" * 60)

        test_data, train_list = load_and_split_dataset(cfg)

        # IMPORTANT: materialize inputs once (avoid iterator exhaustion & allow len())
        test_input_list = list(test_data.input)

        # Fresh memory forecaster per dataset
        memory_forecaster = MemoryAugmentedForecasterV2(memory_config)

        # Build memory from TRAIN ONLY (leakage-safe)
        train_by_id = build_memory_bank(
            train_list=train_list,
            memory_forecaster=memory_forecaster,
            d_model=d_model,
            device=device_obj,
            batch_size=batch_size,
            max_series=max_memory_size,
        )

        # 1) BASELINE
        logger.info("  [baseline]")
        t0 = time.time()
        forecasts = generate_forecasts_baseline(
            test_input_list, pipeline, prediction_length, batch_size, cross_learning=False
        )
        metrics = evaluate_forecasts(
            forecasts,
            test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        baseline_mase = float(metrics[0].get(MASE_KEY, 0.0))
        baseline_wql = float(metrics[0].get(WQL_KEY, 0.0))
        logger.info(f"    MASE={baseline_mase:.4f}, WQL={baseline_wql:.4f} ({time.time()-t0:.1f}s)")
        all_results.append({"dataset": dataset_name, "model": "baseline", "MASE": baseline_mase, "WQL": baseline_wql})

        # 2) CROSS_LEARNING
        logger.info("  [cross_learning]")
        t0 = time.time()
        forecasts = generate_forecasts_baseline(
            test_input_list, pipeline, prediction_length, batch_size, cross_learning=True
        )
        metrics = evaluate_forecasts(
            forecasts,
            test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        cl_mase = float(metrics[0].get(MASE_KEY, 0.0))
        cl_wql = float(metrics[0].get(WQL_KEY, 0.0))
        logger.info(f"    MASE={cl_mase:.4f}, WQL={cl_wql:.4f} ({time.time()-t0:.1f}s)")
        all_results.append({"dataset": dataset_name, "model": "cross_learning", "MASE": cl_mase, "WQL": cl_wql})

        # 3) MEMORY (Option A)
        logger.info(
            "  [memory_augmented_option_a] "
            f"(retrieve_k={retrieve_k}, use_k={use_k}, sim_thr={similarity_threshold}, gap={gap_threshold}, "
            f"min_train={min_train_series_for_memory})"
        )
        t0 = time.time()
        forecasts, diag = generate_forecasts_with_memory_option_a(
            test_input_list,
            pipeline,
            memory_forecaster,
            train_by_id=train_by_id,
            prediction_length=prediction_length,
            batch_size=batch_size,
            retrieve_k=retrieve_k,
            use_k=use_k,
            similarity_threshold=similarity_threshold,
            gap_threshold=gap_threshold,
            min_train_series_for_memory=min_train_series_for_memory,
        )
        metrics = evaluate_forecasts(
            forecasts,
            test_data=test_data,
            metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
            batch_size=5000,
        ).reset_index(drop=True).to_dict(orient="records")
        mem_mase = float(metrics[0].get(MASE_KEY, 0.0))
        mem_wql = float(metrics[0].get(WQL_KEY, 0.0))

        logger.info(f"    MASE={mem_mase:.4f}, WQL={mem_wql:.4f} ({time.time()-t0:.1f}s)")
        if diag.get("disabled", False):
            logger.info(f"    Memory disabled: {diag.get('disabled_reason','')}")
        logger.info(f"    Candidate rate: {diag['candidate_rate']:.1%}, Gate rate: {diag['gate_rate']:.1%}")
        logger.info(f"    Avg neighbors used (gated only): {diag['avg_neighbors_used']:.2f}")
        logger.info(
            f"    Max sim: mean={diag['max_sim_mean']:.4f}, p50={diag['max_sim_p50']:.4f}, "
            f"p90={diag['max_sim_p90']:.4f}, max={diag['max_sim_max']:.4f}"
        )
        logger.info(
            f"    Gap(top1-top2): mean={diag['gap_mean']:.4f}, p50={diag['gap_p50']:.4f}, p90={diag['gap_p90']:.4f}"
        )

        all_results.append(
            {
                "dataset": dataset_name,
                "model": "memory_augmented_option_a",
                "MASE": mem_mase,
                "WQL": mem_wql,
                "candidate_rate": diag["candidate_rate"],
                "gate_rate": diag["gate_rate"],
                "avg_neighbors_used": diag["avg_neighbors_used"],
                "max_sim_mean": diag["max_sim_mean"],
                "max_sim_p90": diag["max_sim_p90"],
                "gap_mean": diag["gap_mean"],
                "gap_p90": diag["gap_p90"],
                "memory_disabled": bool(diag.get("disabled", False)),
                "memory_disabled_reason": diag.get("disabled_reason", ""),
            }
        )

        logger.info(f"  Summary for {dataset_name}:")
        if baseline_mase != 0:
            # This reports "relative improvement" where + means BETTER (lower MASE),
            # and - means WORSE, because MASE lower is better.
            cl_impr = (baseline_mase - cl_mase) / baseline_mase * 100
            mem_impr = (baseline_mase - mem_mase) / baseline_mase * 100
            logger.info(f"    cross_learning vs baseline: {cl_impr:+.2f}% (positive=better)")
            logger.info(f"    memory(A) vs baseline: {mem_impr:+.2f}% (positive=better)")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    pivot = df.pivot(index="dataset", columns="model", values="MASE")
    cols = [c for c in ["baseline", "cross_learning", "memory_augmented_option_a"] if c in pivot.columns]
    pivot = pivot[cols]
    logger.info("\nMASE Results:")
    logger.info(pivot.to_string())

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    app()