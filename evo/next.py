"""
Post-experiment script. Run after EVERY training run.

Handles: result logging, weight updates, and scheduling the next action.
Prints the next action (crossover or tune) with pre-sampled parents so
the agent cannot skip the scheduler.

Usage:
  # After a successful run:
  python evo/next.py --parents "mamba,baseline_v0" --val_bpb 0.9850 --best_bpb 0.9900 --desc "added selective SSM layer"

  # After a crash or skip (still advances the scheduler):
  python evo/next.py --parents "mamba,baseline_v0" --crash --desc "OOM on SSM kernel"

  # First run (baseline, no parents):
  python evo/next.py --baseline --val_bpb 0.9980
"""

import argparse
import sys
from pathlib import Path

# Add parent dir so we can import evo modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from evo.crossover import (
    load_pool,
    save_pool,
    update_weights,
    register_winner,
    log_loss,
    should_crossover,
    sample_parents,
    get_parent_info,
    show_pool,
)


def main():
    parser = argparse.ArgumentParser(description="Post-experiment: log result + schedule next action")
    parser.add_argument("--parents", type=str, help="Comma-separated parent IDs (e.g. 'mamba,baseline_v0')")
    parser.add_argument("--val_bpb", type=float, help="Achieved val_bpb (omit for crashes)")
    parser.add_argument("--best_bpb", type=float, help="Best val_bpb before this experiment")
    parser.add_argument("--desc", type=str, default="", help="Short description of what was tried")
    parser.add_argument("--crash", action="store_true", help="Flag this as a crashed run")
    parser.add_argument("--baseline", action="store_true", help="Record baseline (no parents, no weight update)")
    args = parser.parse_args()

    pool = load_pool()

    # ── Record the result ────────────────────────────────────────────
    print("=" * 60)
    print("  EXPERIMENT RESULT")
    print("=" * 60)

    if args.baseline:
        if args.val_bpb is None:
            print("ERROR: --baseline requires --val_bpb")
            sys.exit(1)
        pool["entries"]["baseline_v0"]["val_bpb"] = args.val_bpb
        save_pool(pool)
        print(f"  Baseline recorded: val_bpb = {args.val_bpb:.6f}")
        print(f"  Description: {args.desc or 'baseline run'}")

    elif args.crash:
        parent_ids = tuple(args.parents.split(",")) if args.parents else ("unknown", "unknown")
        pool = log_loss(pool, parent_ids, None)
        save_pool(pool)
        print(f"  CRASH — logged as loss")
        print(f"  Parents: {parent_ids[0]} x {parent_ids[1]}")
        print(f"  Description: {args.desc}")

    elif args.parents and args.val_bpb is not None and args.best_bpb is not None:
        parent_ids = tuple(args.parents.split(","))
        delta_bpb = args.best_bpb - args.val_bpb
        won = delta_bpb > 0

        # Update parent weights
        pool = update_weights(pool, parent_ids, won, delta_bpb, args.best_bpb)

        if won:
            # Register winner
            train_py = Path("train.py").read_text()
            pool = register_winner(pool, parent_ids, args.val_bpb, args.best_bpb, args.desc)
            print(f"  WIN — val_bpb: {args.val_bpb:.6f} (improved by {delta_bpb:.6f})")
            print(f"  Parents: {parent_ids[0]} x {parent_ids[1]}")
            print(f"  Description: {args.desc}")
            print(f"  Offspring added to gene pool")
        else:
            pool = log_loss(pool, parent_ids, args.val_bpb)
            print(f"  LOSS — val_bpb: {args.val_bpb:.6f} (worse by {abs(delta_bpb):.6f})")
            print(f"  Parents: {parent_ids[0]} x {parent_ids[1]}")
            print(f"  Description: {args.desc}")
            print(f"  Revert train.py to previous commit")

        save_pool(pool)
    else:
        print("ERROR: Provide --baseline, --crash, or (--parents + --val_bpb + --best_bpb)")
        sys.exit(1)

    # ── Schedule next action ─────────────────────────────────────────
    pool = load_pool()  # reload after save
    decision = should_crossover(pool)

    print()
    print("=" * 60)
    print("  NEXT ITERATION")
    print("=" * 60)
    print(f"  Experiment #{decision['experiment']} | "
          f"xover_prob={decision['base_prob']:.3f} | "
          f"consecutive_losses={decision['consecutive_losses']}")
    print()

    if decision["action"] == "crossover":
        print(f"  >>> ACTION: CROSSOVER <<<")
        print(f"  Reason: {decision['reason']}")
        print()

        # Pre-sample parents so the agent can't skip this step
        try:
            a, b = sample_parents(pool)
            ia = get_parent_info(pool, a)
            ib = get_parent_info(pool, b)

            print(f"  Parent A: {a}")
            print(f"    Type: {ia['type']}")
            print(f"    Summary: {ia['summary'][:120]}")
            print(f"    Content: {ia['content_path']}")
            print()
            print(f"  Parent B: {b}")
            print(f"    Type: {ib['type']}")
            print(f"    Summary: {ib['summary'][:120]}")
            print(f"    Content: {ib['content_path']}")
            print()
            print(f"  INSTRUCTION: Spawn a subagent to read both parent files")
            print(f"  and the current train.py, then write a new train.py that")
            print(f"  blends ideas from both parents. Use parent IDs: {a},{b}")
        except ValueError as e:
            print(f"  Cannot sample parents: {e}")
            print(f"  Falling back to direct edit (tune)")
    else:
        print(f"  >>> ACTION: TUNE <<<")
        print(f"  Reason: {decision['reason']}")
        print()
        print(f"  INSTRUCTION: Make a targeted edit to train.py — tweak")
        print(f"  hyperparameters, optimizer settings, or small architectural")
        print(f"  changes based on recent results. Use your own judgment.")

    print()
    print("-" * 60)


if __name__ == "__main__":
    main()
