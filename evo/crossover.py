"""
Crossover engine for evolutionary autoresearch.

This module handles gene pool data management and scheduling. The creative
work of blending parents into a new train.py is done by the agent (or a
subagent it spawns to avoid context bloat from large paper markdown files).

Functions for the agent to call:
- should_crossover: query the scheduler — "crossover" or "tune" this iteration?
- load_pool / save_pool: read/write gene_pool.json
- sample_parents: weighted selection of two parents
- get_parent_info: get metadata + content paths for sampled parents
- update_weights: adjust parent weights after an experiment
- register_winner: add winning offspring to the pool
- log_loss: record a failed experiment in history
- show_pool: pretty-print current pool state
"""

import json
import math
import random
import time
from pathlib import Path

EVO_DIR = Path(__file__).parent
GENE_POOL_PATH = EVO_DIR / "gene_pool.json"
SNAPSHOTS_DIR = EVO_DIR / "snapshots"


def load_pool(path: str | Path | None = None) -> dict:
    """Load the gene pool from disk."""
    p = Path(path) if path else GENE_POOL_PATH
    return json.loads(p.read_text())


def save_pool(pool: dict, path: str | Path | None = None):
    """Save the gene pool to disk."""
    p = Path(path) if path else GENE_POOL_PATH
    p.write_text(json.dumps(pool, indent=2))


def _renormalize(pool: dict):
    """Renormalize all weights to sum to 1.0."""
    total = sum(e["weight"] for e in pool["entries"].values())
    if total > 0:
        for entry in pool["entries"].values():
            entry["weight"] /= total


def _apply_floor(pool: dict):
    """Ensure no weight drops below min_weight."""
    min_w = pool["hyperparams"]["min_weight"]
    for entry in pool["entries"].values():
        entry["weight"] = max(entry["weight"], min_w)
    _renormalize(pool)


# ── Scheduler ────────────────────────────────────────────────────────

# Default schedule config — merged into gene_pool.json on first use.
DEFAULT_SCHEDULE = {
    "type": "cosine",       # "constant", "linear", "exp", "cosine", "step"
    "xover_start": 0.80,    # crossover probability at experiment 0
    "xover_end": 0.05,      # crossover probability at experiment n_total
    "n_total": 200,         # expected total experiments (for schedule progress)
    "staleness_trigger": 5, # force crossover after N consecutive losses (0=off)
}


def _schedule_prob(step: int, cfg: dict) -> float:
    """Compute base crossover probability at a given step."""
    t = step / max(cfg["n_total"] - 1, 1)  # 0..1
    t = min(t, 1.0)
    s, e = cfg["xover_start"], cfg["xover_end"]
    stype = cfg["type"]

    if stype == "constant":
        return s
    elif stype == "linear":
        return s + (e - s) * t
    elif stype == "exp":
        if s <= 0 or e <= 0:
            return s + (e - s) * t
        return s * (e / s) ** t
    elif stype == "cosine":
        return e + (s - e) * 0.5 * (1 + math.cos(math.pi * t))
    elif stype == "step":
        return s if t < 0.5 else e
    return s


def _consecutive_losses(pool: dict) -> int:
    """Count consecutive losses from the end of history."""
    count = 0
    for h in reversed(pool.get("history", [])):
        if h["won"]:
            break
        count += 1
    return count


def should_crossover(pool: dict) -> dict:
    """
    Query the scheduler: should this iteration be a crossover or a tune?

    Reads schedule config from pool["schedule"] (falls back to defaults).
    Uses pool["history"] length as the experiment counter.

    Returns:
        {
            "action": "crossover" | "tune",
            "reason": str,          # human-readable explanation
            "experiment": int,      # current experiment number
            "base_prob": float,     # schedule probability (before overrides)
            "consecutive_losses": int,
        }
    """
    cfg = {**DEFAULT_SCHEDULE, **pool.get("schedule", {})}
    step = len(pool.get("history", []))
    base_prob = _schedule_prob(step, cfg)
    consec = _consecutive_losses(pool)

    # Staleness override: force crossover if stuck
    if cfg["staleness_trigger"] > 0 and consec >= cfg["staleness_trigger"]:
        return {
            "action": "crossover",
            "reason": f"staleness: {consec} consecutive losses (trigger={cfg['staleness_trigger']})",
            "experiment": step,
            "base_prob": base_prob,
            "consecutive_losses": consec,
        }

    # Roll against schedule probability
    roll = random.random()
    if roll < base_prob:
        return {
            "action": "crossover",
            "reason": f"schedule: rolled {roll:.3f} < {base_prob:.3f} "
                      f"({cfg['type']} at step {step}/{cfg['n_total']})",
            "experiment": step,
            "base_prob": base_prob,
            "consecutive_losses": consec,
        }
    else:
        return {
            "action": "tune",
            "reason": f"schedule: rolled {roll:.3f} >= {base_prob:.3f} "
                      f"({cfg['type']} at step {step}/{cfg['n_total']})",
            "experiment": step,
            "base_prob": base_prob,
            "consecutive_losses": consec,
        }


def sample_parents(pool: dict) -> tuple[str, str]:
    """
    Sample two distinct parents from the pool, weighted by fitness.
    Returns (parent_a_id, parent_b_id).
    """
    entries = pool["entries"]
    ids = list(entries.keys())
    weights = [entries[i]["weight"] for i in ids]

    if len(ids) < 2:
        raise ValueError(f"Need at least 2 entries in pool, got {len(ids)}")

    parent_a = random.choices(ids, weights=weights, k=1)[0]

    remaining_ids = [i for i in ids if i != parent_a]
    remaining_weights = [entries[i]["weight"] for i in remaining_ids]
    parent_b = random.choices(remaining_ids, weights=remaining_weights, k=1)[0]

    return parent_a, parent_b


def get_parent_info(pool: dict, parent_id: str) -> dict:
    """
    Get metadata and content path for a parent entry.

    Returns a dict with:
      - id: the parent ID
      - type: "paper" or "code"
      - summary: short description (safe to read into main agent context)
      - content_path: absolute path to full content file (for subagent to read)
      - val_bpb: measured performance (code entries only, None for papers)
      - parents: list of parent IDs (if offspring), None for seeds
    """
    entry = pool["entries"][parent_id]
    content_path = EVO_DIR / entry["content_path"]

    return {
        "id": parent_id,
        "type": entry["type"],
        "summary": entry.get("summary", ""),
        "content_path": str(content_path),
        "val_bpb": entry.get("val_bpb"),
        "parents": entry.get("parents"),
    }


def update_weights(
    pool: dict,
    parent_ids: tuple[str, str],
    won: bool,
    delta_bpb: float,
    best_bpb: float,
) -> dict:
    """
    Update parent weights based on experiment outcome.

    Args:
        pool: gene pool dict
        parent_ids: (parent_a_id, parent_b_id)
        won: True if offspring beat the current best
        delta_bpb: best_bpb - new_bpb (positive = improvement)
        best_bpb: current best val_bpb before this experiment
    """
    alpha = pool["hyperparams"]["alpha"]
    beta = pool["hyperparams"]["beta"]

    for pid in parent_ids:
        if pid not in pool["entries"]:
            continue
        if won:
            boost = 1 + alpha * (delta_bpb / best_bpb)
            pool["entries"][pid]["weight"] *= boost
        else:
            penalty = 1 - beta * min(abs(delta_bpb) / best_bpb, 1.0)
            pool["entries"][pid]["weight"] *= max(penalty, 0.01)

    _apply_floor(pool)
    return pool


def register_winner(
    pool: dict,
    parent_ids: tuple[str, str],
    train_py_content: str,
    val_bpb: float,
    best_bpb: float,
    description: str,
) -> dict:
    """
    Add a winning offspring to the gene pool.

    The offspring's initial weight is earned, not inherited — it starts at
    pool-average scaled by the magnitude of the improvement.
    """
    alpha = pool["hyperparams"]["alpha"]

    # Generate unique ID
    generation = max((e.get("generation", 0) for e in pool["entries"].values()), default=0) + 1
    offspring_id = f"offspring_{generation}"

    # Save code snapshot to disk
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    snapshot_path = SNAPSHOTS_DIR / f"{offspring_id}.py"
    snapshot_path.write_text(train_py_content)

    # Compute initial weight: pool-average scaled by improvement
    n_entries = len(pool["entries"]) + 1
    base_weight = 1.0 / n_entries
    delta_bpb = best_bpb - val_bpb
    bonus = alpha * (delta_bpb / best_bpb)
    initial_weight = base_weight * (1 + bonus)

    pool["entries"][offspring_id] = {
        "type": "code",
        "summary": description,
        "content_path": str(snapshot_path.relative_to(EVO_DIR)),
        "val_bpb": val_bpb,
        "parents": list(parent_ids),
        "weight": initial_weight,
        "generation": generation,
    }

    pool["history"].append({
        "offspring_id": offspring_id,
        "parents": list(parent_ids),
        "val_bpb": val_bpb,
        "won": True,
        "timestamp": time.time(),
    })

    _apply_floor(pool)
    return pool


def log_loss(pool: dict, parent_ids: tuple[str, str], val_bpb: float | None) -> dict:
    """Log a losing experiment to history (for learning, not added to pool)."""
    pool["history"].append({
        "offspring_id": None,
        "parents": list(parent_ids),
        "val_bpb": val_bpb,
        "won": False,
        "timestamp": time.time(),
    })
    return pool


def show_pool(pool: dict):
    """Pretty-print the gene pool state."""
    print(f"Gene Pool ({len(pool['entries'])} entries)")
    print(f"Hyperparams: alpha={pool['hyperparams']['alpha']}, "
          f"beta={pool['hyperparams']['beta']}, "
          f"min_weight={pool['hyperparams']['min_weight']}")

    cfg = {**DEFAULT_SCHEDULE, **pool.get("schedule", {})}
    step = len(pool.get("history", []))
    prob = _schedule_prob(step, cfg)
    consec = _consecutive_losses(pool)
    print(f"Schedule: {cfg['type']} {cfg['xover_start']}→{cfg['xover_end']} "
          f"over {cfg['n_total']} exps, staleness@{cfg['staleness_trigger']}")
    print(f"  Step {step}/{cfg['n_total']}, "
          f"current xover_prob={prob:.3f}, "
          f"consecutive_losses={consec}")

    print("-" * 70)
    sorted_entries = sorted(pool["entries"].items(), key=lambda x: -x[1]["weight"])
    for eid, entry in sorted_entries:
        bpb = f"bpb={entry['val_bpb']:.4f}" if entry.get("val_bpb") else "bpb=n/a"
        parents = f"parents={entry['parents']}" if entry.get("parents") else "seed"
        print(f"  {entry['weight']:.4f}  {entry['type']:5s}  {eid:20s}  {bpb:12s}  {parents}")
    print(f"\nHistory: {len(pool['history'])} experiments logged")
    wins = sum(1 for h in pool["history"] if h["won"])
    print(f"  Wins: {wins}, Losses: {len(pool['history']) - wins}")
