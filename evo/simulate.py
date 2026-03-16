"""
Simulate evolutionary search dynamics to test hyperparameters.

Models a realistic landscape with many architecture basins — most are
dead ends or marginal, with only a few good ones. Tests whether the
gene pool weight dynamics + crossover can escape suboptimal wells.

Based on real data from autoresearch progress chart:
- Start: 0.998 BPB
- Current best after 83 exps: ~0.977
- Global best ever achieved: 0.9358
- Win rate: ~18% (15/83)
- Typical improvement magnitudes: 0.001–0.008 BPB
"""

import random
import math
from collections import defaultdict

# ── Architecture basins (wells) ──────────────────────────────────────
# Realistic landscape: many basins, most are traps (worse or marginally
# better than current). Only a few lead to genuinely better architectures.
BASINS = {
    # ── Current basin (where you start) ──
    "gpt_attn":     {"floor": 0.974, "entry_bpb": 0.998, "quality": "start"},

    # ── Dead ends / traps (worse or barely equal to current) ──
    "rwkv_pure":    {"floor": 0.980, "entry_bpb": 0.995, "quality": "trap"},
    "retnet_pure":  {"floor": 0.978, "entry_bpb": 0.996, "quality": "trap"},
    "hyena_pure":   {"floor": 0.982, "entry_bpb": 0.997, "quality": "trap"},
    "conv_mixer":   {"floor": 0.985, "entry_bpb": 0.999, "quality": "trap"},
    "moe_naive":    {"floor": 0.976, "entry_bpb": 0.993, "quality": "trap"},
    "mega":         {"floor": 0.977, "entry_bpb": 0.994, "quality": "trap"},
    "aft":          {"floor": 0.979, "entry_bpb": 0.996, "quality": "trap"},

    # ── Marginal improvements (slightly better than current) ──
    "gla_hybrid":   {"floor": 0.968, "entry_bpb": 0.992, "quality": "marginal"},
    "retnet_hyb":   {"floor": 0.970, "entry_bpb": 0.993, "quality": "marginal"},
    "xlstm_hyb":    {"floor": 0.971, "entry_bpb": 0.991, "quality": "marginal"},

    # ── Good basins (meaningfully better) ──
    "ssm_hybrid":   {"floor": 0.955, "entry_bpb": 0.990, "quality": "good"},
    "linear_attn":  {"floor": 0.960, "entry_bpb": 0.992, "quality": "good"},

    # ── Global optimum (hard to find) ──
    "deep_ssm":     {"floor": 0.938, "entry_bpb": 0.985, "quality": "optimal"},
}

N_BASINS = len(BASINS)

# ── Simulation parameters ────────────────────────────────────────────
N_EXPERIMENTS = 200
N_TRIALS = 1000  # Monte Carlo trials per configuration


def simulate_basin_improvement(current_bpb: float, basin: dict, steps_in_basin: int) -> float:
    """
    Simulate one hyperparameter tuning step within a basin.
    Calibrated to real autoresearch data with three phases:

    Phase 1 (steps 0-10): Early burst — ~80% win rate, large improvements.
      The easy wins: batch size, warmdown, warmup, depth, init scale.
      Real data: 8/10 wins, improvements 0.002–0.008 BPB.

    Phase 2 (steps 10-40): Plateau — ~12% win rate, moderate improvements.
      Harder tweaks: LR, window patterns, embedding LR.
      Real data: 5/30 wins, improvements 0.001–0.003 BPB.

    Phase 3 (steps 40+): Squeeze — ~5% win rate, tiny improvements.
      Scraping the barrel: RoPE frequency, random seeds.
      Real data: 2/40 wins, improvements 0.0005–0.001 BPB.
    """
    floor = basin["floor"]
    gap = current_bpb - floor
    entry_gap = basin["entry_bpb"] - floor

    if gap <= 0.0001:
        return current_bpb + random.uniform(0, 0.003)

    # Three phases based on steps in this basin
    if steps_in_basin < 10:
        # Phase 1: Early burst
        win_prob = 0.80 * (gap / entry_gap) if entry_gap > 0 else 0.30
        win_prob = max(win_prob, 0.30)
        max_improvement = gap * 0.20  # can close up to 20% of gap
        min_improvement = gap * 0.05  # at least 5% of gap when winning
    elif steps_in_basin < 40:
        # Phase 2: Plateau
        win_prob = 0.12 * (gap / entry_gap) if entry_gap > 0 else 0.04
        win_prob = max(win_prob, 0.04)
        max_improvement = gap * 0.08
        min_improvement = gap * 0.01
    else:
        # Phase 3: Squeeze
        win_prob = 0.05 * (gap / entry_gap) if entry_gap > 0 else 0.02
        win_prob = max(win_prob, 0.02)
        max_improvement = gap * 0.04
        min_improvement = 0.0001

    if random.random() < win_prob:
        improvement = random.uniform(min_improvement, max_improvement)
        return current_bpb - improvement
    else:
        return current_bpb + random.uniform(0, 0.005)


def attempt_crossover(current_basin: str, current_bpb: float) -> tuple[str, float]:
    """
    Simulate a genetic crossover attempt.

    Realistic assumptions:
    - 40% chance: stays in current basin (incremental blend)
    - 60% chance: jumps to a random other basin

    Within-basin crossover is WORSE than targeted hyperparameter tuning:
    - Random blend of two similar architectures is clumsy compared to
      an LLM surgically tweaking learning rate or window size
    - Lower win rate, smaller improvements, risk of regression

    Basin jumps have real costs:
    - 30% chance the blended architecture is broken (won't train,
      NaN loss, incompatible components)
    - When it works, you start partway through the new basin's
      optimization curve (carryover from shared ideas)
    """
    if random.random() < 0.40:
        # Same basin — crossover is a clumsy version of tuning
        # Worse than phase 2 tuning: lower win rate, smaller max improvement
        basin = BASINS[current_basin]
        gap = current_bpb - basin["floor"]
        if gap <= 0.0001:
            return current_basin, current_bpb + random.uniform(0, 0.003)

        win_prob = 0.08  # much worse than targeted tuning (~12% phase 2)
        max_improvement = gap * 0.04  # at most 4% gap closure (vs 8% for tuning)

        if random.random() < win_prob:
            improvement = random.uniform(0.0001, max_improvement)
            return current_basin, current_bpb - improvement
        else:
            # Crossover blend often makes things worse
            return current_basin, current_bpb + random.uniform(0, 0.008)
    else:
        # Basin jump attempt
        other_basins = [b for b in BASINS if b != current_basin]
        new_basin = random.choice(other_basins)
        new_data = BASINS[new_basin]
        entry = new_data["entry_bpb"]
        floor = new_data["floor"]

        # 30% chance the blended architecture is broken
        # (incompatible components, NaN loss, won't compile)
        if random.random() < 0.30:
            return new_basin, current_bpb + random.uniform(0.01, 0.03)

        # When it works, carry over 20-50% of optimization progress
        carryover = random.uniform(0.20, 0.50)
        new_bpb = entry - carryover * (entry - floor)
        return new_basin, new_bpb


def _annealing_crossover_prob(
    step: int,
    n_total: int,
    schedule: str,
    xover_start: float,
    xover_end: float,
) -> float:
    """
    Compute crossover probability at a given step under an annealing schedule.

    Schedules:
    - "constant": flat probability (xover_start used, xover_end ignored)
    - "linear":   linear decay from xover_start → xover_end
    - "exp":      exponential decay from xover_start → xover_end
    - "cosine":   cosine annealing from xover_start → xover_end
    - "step":     high for first half, low for second half
    """
    if schedule == "constant":
        return xover_start

    t = step / max(n_total - 1, 1)  # 0..1

    if schedule == "linear":
        return xover_start + (xover_end - xover_start) * t
    elif schedule == "exp":
        # Exponential decay: start * (end/start)^t
        if xover_start <= 0 or xover_end <= 0:
            return xover_start * (1 - t) + xover_end * t  # fallback to linear
        return xover_start * (xover_end / xover_start) ** t
    elif schedule == "cosine":
        # Cosine annealing
        return xover_end + (xover_start - xover_end) * 0.5 * (1 + math.cos(math.pi * t))
    elif schedule == "step":
        # Step function: high for first half, low for second
        return xover_start if t < 0.5 else xover_end
    else:
        return xover_start


def run_simulation(
    alpha: float,
    beta: float,
    min_weight: float,
    crossover_every: int,       # Force crossover every N iterations (0 = never)
    crossover_prob: float,      # Per-iteration probability of crossover (used for constant)
    staleness_trigger: int,     # Force crossover after N consecutive losses (0 = disabled)
    schedule: str = "constant", # Annealing schedule
    xover_start: float = 0.0,   # Starting crossover prob (for annealing)
    xover_end: float = 0.0,     # Ending crossover prob (for annealing)
    n_experiments: int = N_EXPERIMENTS,
) -> dict:
    """Run one simulation trial."""

    current_basin = "gpt_attn"
    current_bpb = BASINS[current_basin]["entry_bpb"]
    best_bpb = current_bpb
    consecutive_losses = 0
    steps_in_basin = 0

    trajectory = []
    wins = 0
    crossovers_done = 0
    basin_jumps = 0
    early_burst_wins = 0

    for i in range(n_experiments):
        # Compute crossover probability for this step
        if schedule != "constant":
            p_xover = _annealing_crossover_prob(i, n_experiments, schedule, xover_start, xover_end)
        else:
            p_xover = crossover_prob

        do_crossover = False
        if crossover_every > 0 and (i + 1) % crossover_every == 0:
            do_crossover = True
        elif p_xover > 0 and random.random() < p_xover:
            do_crossover = True
        if staleness_trigger > 0 and consecutive_losses >= staleness_trigger:
            do_crossover = True

        if do_crossover:
            crossovers_done += 1
            new_basin, new_bpb = attempt_crossover(current_basin, current_bpb)

            if new_bpb < best_bpb:
                best_bpb = new_bpb
                current_bpb = new_bpb
                if new_basin != current_basin:
                    basin_jumps += 1
                    steps_in_basin = 0  # reset tuning curve in new basin
                current_basin = new_basin
                consecutive_losses = 0
                wins += 1
                trajectory.append(("xwin", i, new_bpb, new_basin))
            else:
                consecutive_losses += 1
                trajectory.append(("xloss", i, new_bpb, new_basin))
        else:
            new_bpb = simulate_basin_improvement(
                current_bpb, BASINS[current_basin], steps_in_basin
            )
            steps_in_basin += 1

            if new_bpb < best_bpb:
                if steps_in_basin <= 10:
                    early_burst_wins += 1
                best_bpb = new_bpb
                current_bpb = new_bpb
                consecutive_losses = 0
                wins += 1
                trajectory.append(("twin", i, new_bpb, current_basin))
            else:
                consecutive_losses += 1
                trajectory.append(("tloss", i, new_bpb, current_basin))

    quality = BASINS[current_basin]["quality"]
    return {
        "final_bpb": best_bpb,
        "final_basin": current_basin,
        "basin_quality": quality,
        "wins": wins,
        "early_burst_wins": early_burst_wins,
        "crossovers_done": crossovers_done,
        "basin_jumps": basin_jumps,
        "found_optimal": quality == "optimal",
        "found_good_or_better": quality in ("good", "optimal"),
        "stuck_in_trap": quality in ("trap", "start"),
    }


def run_monte_carlo(config: dict, n_trials: int = N_TRIALS) -> dict:
    """Run many trials and aggregate stats."""
    results = [run_simulation(**config, n_experiments=N_EXPERIMENTS) for _ in range(n_trials)]

    final_bpbs = [r["final_bpb"] for r in results]
    final_bpbs.sort()

    basin_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    for r in results:
        basin_counts[r["final_basin"]] += 1
        quality_counts[r["basin_quality"]] += 1

    return {
        "config": config,
        "n_trials": n_trials,
        "bpb_mean": sum(final_bpbs) / len(final_bpbs),
        "bpb_median": final_bpbs[len(final_bpbs) // 2],
        "bpb_min": min(final_bpbs),
        "bpb_max": max(final_bpbs),
        "bpb_p10": final_bpbs[int(0.1 * n_trials)],
        "bpb_p90": final_bpbs[int(0.9 * n_trials)],
        "pct_optimal": sum(1 for r in results if r["found_optimal"]) / n_trials * 100,
        "pct_good_plus": sum(1 for r in results if r["found_good_or_better"]) / n_trials * 100,
        "pct_stuck": sum(1 for r in results if r["stuck_in_trap"]) / n_trials * 100,
        "avg_wins": sum(r["wins"] for r in results) / n_trials,
        "avg_early_burst": sum(r["early_burst_wins"] for r in results) / n_trials,
        "avg_crossovers": sum(r["crossovers_done"] for r in results) / n_trials,
        "avg_basin_jumps": sum(r["basin_jumps"] for r in results) / n_trials,
        "quality_pct": {q: c / n_trials * 100 for q, c in quality_counts.items()},
        "top_basins": dict(sorted(basin_counts.items(), key=lambda x: -x[1])[:6]),
    }


def fmt(label: str, s: dict) -> str:
    c = s["config"]
    sched = c.get("schedule", "constant")
    sched_str = f"  schedule={sched}" if sched != "constant" else ""
    if sched != "constant":
        sched_str += f"  xover: {c.get('xover_start', '?')}→{c.get('xover_end', '?')}"
    lines = [
        f"\n{'='*75}",
        f"  {label}",
        f"{'='*75}",
        f"  alpha={c['alpha']}  beta={c['beta']}  min_w={c['min_weight']}  "
        f"stale={c['staleness_trigger']}{sched_str}",
        f"{'─'*75}",
        f"  BPB   mean={s['bpb_mean']:.4f}  med={s['bpb_median']:.4f}  "
        f"min={s['bpb_min']:.4f}  p10={s['bpb_p10']:.4f}  p90={s['bpb_p90']:.4f}",
        f"  Found optimal basin: {s['pct_optimal']:5.1f}%   "
        f"Good+: {s['pct_good_plus']:5.1f}%   "
        f"Stuck in trap: {s['pct_stuck']:5.1f}%",
        f"  Avg wins: {s['avg_wins']:.1f}/{N_EXPERIMENTS}   "
        f"early burst wins: {s['avg_early_burst']:.1f}   "
        f"crossovers: {s['avg_crossovers']:.1f}   "
        f"basin jumps: {s['avg_basin_jumps']:.1f}",
        f"  Quality: {' | '.join(f'{q}={p:.1f}%' for q,p in sorted(s['quality_pct'].items()))}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    random.seed(42)

    print("=" * 75)
    print("  EVOLUTIONARY SEARCH SIMULATION (realistic landscape)")
    print(f"  {N_EXPERIMENTS} experiments/trial, {N_TRIALS} trials, {N_BASINS} basins")
    print(f"  Traps: 8 | Marginal: 3 | Good: 2 | Optimal: 1")
    print(f"  BPB floors — traps: 0.976-0.985, marginal: 0.968-0.971, "
          f"good: 0.955-0.960, optimal: 0.938")
    print("=" * 75)

    base = {"alpha": 1.0, "beta": 0.3, "min_weight": 0.02, "crossover_every": 0, "crossover_prob": 0.0}

    configs = [
        # ── Controls: constant crossover rates ───────────────────────
        ("CONTROL: Pure greedy (no crossover)", {
            **base, "staleness_trigger": 0,
        }),
        ("CONST 20% + stale@5", {
            **base, "crossover_prob": 0.20, "staleness_trigger": 5,
        }),
        ("CONST 50% + stale@5", {
            **base, "crossover_prob": 0.50, "staleness_trigger": 5,
        }),
        ("CONST 80% + stale@5", {
            **base, "crossover_prob": 0.80, "staleness_trigger": 5,
        }),
        ("CONST 100% (all crossover, no tuning)", {
            **base, "crossover_prob": 1.00, "staleness_trigger": 0,
        }),
        ("CONST 80%, NO staleness", {
            **base, "crossover_prob": 0.80, "staleness_trigger": 0,
        }),
        ("CONST 100% + stale@5", {
            **base, "crossover_prob": 1.00, "staleness_trigger": 5,
        }),

        # ── Best annealing configs from previous run ─────────────────
        ("STEP 80%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "step", "xover_start": 0.80, "xover_end": 0.05,
        }),
        ("LINEAR 90%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "linear", "xover_start": 0.90, "xover_end": 0.05,
        }),
        ("COSINE 80%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "cosine", "xover_start": 0.80, "xover_end": 0.05,
        }),
        ("EXP 90%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "exp", "xover_start": 0.90, "xover_end": 0.05,
        }),

        # ── Annealing from 100% ─────────────────────────────────────
        ("STEP 100%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "step", "xover_start": 1.00, "xover_end": 0.05,
        }),
        ("LINEAR 100%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "linear", "xover_start": 1.00, "xover_end": 0.05,
        }),
        ("COSINE 100%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "cosine", "xover_start": 1.00, "xover_end": 0.05,
        }),
        ("EXP 100%→5% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "exp", "xover_start": 1.00, "xover_end": 0.05,
        }),

        # ── Annealing down to higher floor ───────────────────────────
        ("STEP 80%→20% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "step", "xover_start": 0.80, "xover_end": 0.20,
        }),
        ("COSINE 80%→20% + stale@5", {
            **base, "staleness_trigger": 5,
            "schedule": "cosine", "xover_start": 0.80, "xover_end": 0.20,
        }),
    ]

    all_results = []
    for label, config in configs:
        stats = run_monte_carlo(config)
        print(fmt(label, stats))
        all_results.append((label, stats))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    print(f"  {'Config':<50s} {'Mean':>6s} {'Med':>6s} "
          f"{'%Opt':>5s} {'%Good+':>6s} {'%Stuck':>6s} {'EBurst':>6s} {'Xovers':>6s} {'Jumps':>5s}")
    print("─" * 110)
    for label, s in all_results:
        short = label[:49]
        print(f"  {short:<50s} {s['bpb_mean']:.4f} {s['bpb_median']:.4f} "
              f"{s['pct_optimal']:5.1f} {s['pct_good_plus']:6.1f} "
              f"{s['pct_stuck']:6.1f} {s['avg_early_burst']:6.1f} "
              f"{s['avg_crossovers']:6.1f} {s['avg_basin_jumps']:5.1f}")
