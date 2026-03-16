"""
Generate a crossover train.py by calling Kimi K2 with both parent files.

Reads parent info from evo/.next_action.json (written by next.py), calls
the Groq API, and writes the result directly to train.py.

The agent runs this script — it does not spawn subagents or write code itself.

Usage:
    python evo/gen_crossover.py              # reads .next_action.json
    python evo/gen_crossover.py --parents "mamba,baseline_v0"  # override parents
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

EVO_DIR = Path(__file__).parent
REPO_DIR = EVO_DIR.parent
load_dotenv(EVO_DIR / ".env")

MODEL = "moonshotai/kimi-k2-instruct"
NEXT_ACTION_PATH = EVO_DIR / ".next_action.json"


def load_next_action(parent_override: str | None) -> dict:
    if parent_override:
        a, b = parent_override.split(",")
        from evo.crossover import load_pool, get_parent_info
        pool = load_pool()
        ia = get_parent_info(pool, a.strip())
        ib = get_parent_info(pool, b.strip())
        return {
            "action": "crossover",
            "parent_a": a.strip(), "parent_a_type": ia["type"],
            "parent_a_summary": ia["summary"],
            "parent_a_content": ia["content_path"],
            "parent_b": b.strip(), "parent_b_type": ib["type"],
            "parent_b_summary": ib["summary"],
            "parent_b_content": ib["content_path"],
        }

    if not NEXT_ACTION_PATH.exists():
        print("ERROR: .next_action.json not found. Run evo/next.py first.")
        sys.exit(1)

    action = json.loads(NEXT_ACTION_PATH.read_text())

    if action["action"] != "crossover":
        print(f"ERROR: .next_action.json says '{action['action']}', not crossover.")
        print(f"Reason: {action.get('reason', '?')}")
        print("Edit train.py directly for tune iterations.")
        sys.exit(1)

    return action


def read_content(path: str) -> str:
    p = Path(path)
    if not p.exists():
        # Try relative to repo dir
        p = REPO_DIR / path
    if not p.exists():
        print(f"WARNING: content file not found: {path}")
        return f"[file not found: {path}]"
    return p.read_text()


def build_prompt(action: dict, train_py: str) -> str:
    a_content = read_content(action["parent_a_content"])
    b_content = read_content(action["parent_b_content"])

    a_label = f"{action['parent_a']} ({action['parent_a_type']})"
    b_label = f"{action['parent_b']} ({action['parent_b_type']})"

    # Truncate papers if very long (keep first 60k chars — Kimi K2 has 128k context)
    max_chars = 60_000
    if len(a_content) > max_chars:
        a_content = a_content[:max_chars] + "\n\n[truncated]"
    if len(b_content) > max_chars:
        b_content = b_content[:max_chars] + "\n\n[truncated]"

    return f"""You are an ML researcher doing architecture search. Your task is to blend ideas from two source materials into a new training script.

## Parent A: {a_label}
Summary: {action['parent_a_summary']}

Full content:
{a_content}

---

## Parent B: {b_label}
Summary: {action['parent_b_summary']}

Full content:
{b_content}

---

## Current train.py (base to modify)

{train_py}

---

## Your task

Write a new train.py that blends the most promising architectural ideas from Parent A and Parent B into the current codebase.

Rules:
- Output ONLY the complete Python file, no commentary before or after
- Must import from prepare.py: MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
- Must print val_bpb at the end (the evaluate_bpb call is already there — keep it)
- Must respect TIME_BUDGET for training duration
- No new package dependencies — only what's already imported
- Single file
- Update the MANIFEST comment block at the top to describe the new architecture
- Make coherent changes — don't randomly paste code, pick ideas that fit together
- If a parent is a paper, extract the key mechanism and implement it; don't copy-paste paper text as comments
- If a parent is a code snapshot, you can reuse its patterns directly

Think carefully about which ideas from each parent are most compatible and likely to improve val_bpb. Then write the complete file."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=str, help="Override parents: 'id_a,id_b'")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only, don't call API")
    args = parser.parse_args()

    action = load_next_action(args.parents)

    train_py = (REPO_DIR / "train.py").read_text()

    prompt = build_prompt(action, train_py)

    if args.dry_run:
        print(prompt[:2000])
        print(f"\n[dry-run: prompt is {len(prompt)} chars, would call {MODEL}]")
        return

    print(f"Generating crossover: {action['parent_a']} x {action['parent_b']}")
    print(f"Prompt size: {len(prompt):,} chars")
    print(f"Calling {MODEL}...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8192,
    )

    result = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if result.startswith("```python"):
        result = result[len("```python"):].lstrip("\n")
    if result.startswith("```"):
        result = result[3:].lstrip("\n")
    if result.endswith("```"):
        result = result[:-3].rstrip()

    # Sanity check — must look like Python
    if "def " not in result or "import " not in result:
        print("ERROR: Response doesn't look like valid Python. Aborting.")
        print("First 500 chars:", result[:500])
        sys.exit(1)

    (REPO_DIR / "train.py").write_text(result)

    print(f"\nWrote new train.py ({len(result):,} chars)")
    print(f"Parents: {action['parent_a']} x {action['parent_b']}")
    print(f"\nReview train.py, then:")
    print(f"  git add train.py && git commit -m 'crossover: {action['parent_a']} x {action['parent_b']}'")
    print(f"  uv run train.py > run.log 2>&1")
    print(f"  python evo/next.py --parents '{action['parent_a']},{action['parent_b']}' --val_bpb <BPB> --best_bpb <BEST> --desc '...'")


if __name__ == "__main__":
    main()
