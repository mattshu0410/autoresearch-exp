"""
Extract a paper into the gene pool.
Converts PDF to markdown via docling, generates a short summary via LLM,
and stores the full markdown on disk with a reference in gene_pool.json.

Usage:
    python evo/extract_paper.py --pdf path/to/paper.pdf --name "mamba"
    python evo/extract_paper.py --arxiv 2312.00752 --name "mamba"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from groq import Groq

EVO_DIR = Path(__file__).parent
load_dotenv(EVO_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "moonshotai/kimi-k2-instruct"
GENE_POOL_PATH = EVO_DIR / "gene_pool.json"
PAPERS_DIR = EVO_DIR / "papers"

SUMMARY_PROMPT = """\
Summarize the key architectural innovations from this paper in 3-5 sentences. \
Focus on the core mechanisms, what problems they solve, and what makes this \
architecture unique compared to standard Transformers. Be specific about \
the technical approach."""


def load_gene_pool() -> dict:
    if GENE_POOL_PATH.exists():
        return json.loads(GENE_POOL_PATH.read_text())
    return {
        "schema_version": 1,
        "hyperparams": {"alpha": 1.0, "beta": 0.3, "min_weight": 0.02},
        "entries": {},
        "history": [],
    }


def save_gene_pool(pool: dict):
    GENE_POOL_PATH.write_text(json.dumps(pool, indent=2))


def download_arxiv(arxiv_id: str) -> Path:
    """Download a paper from arxiv by ID."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    dest = PAPERS_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    print(f"Downloading {url}...")
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True, capture_output=True)
    print(f"Saved to {dest}")
    return dest


def convert_pdf_to_md(pdf_path: str) -> str:
    """Convert PDF to markdown using docling."""
    print(f"Converting {pdf_path} to markdown...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()
    print(f"  -> {len(md.split())} words extracted")
    return md


def generate_summary(paper_md: str) -> str:
    """Generate a short summary of the paper's architectural ideas."""
    client = Groq(api_key=GROQ_API_KEY)
    print("Generating summary via LLM...")
    # Use first ~8000 words for summary to stay within token limits
    words = paper_md.split()
    truncated = " ".join(words[:8000]) if len(words) > 8000 else paper_md
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": truncated},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def add_to_pool(name: str, summary: str, md_path: Path):
    """Add a paper entry to the gene pool."""
    pool = load_gene_pool()

    if name in pool["entries"]:
        print(f"Entry '{name}' already exists. Overwriting.")

    n_entries = len(pool["entries"]) + (0 if name in pool["entries"] else 1)
    new_weight = 1.0 / n_entries

    # Renormalize existing entries
    for eid, entry in pool["entries"].items():
        if eid != name:
            entry["weight"] = entry["weight"] * (n_entries - 1) / n_entries

    pool["entries"][name] = {
        "type": "paper",
        "summary": summary,
        "content_path": str(md_path.relative_to(EVO_DIR)),
        "source_pdf": str(md_path.with_suffix(".pdf").relative_to(EVO_DIR)),
        "weight": new_weight,
        "val_bpb": None,
        "parents": None,
        "generation": 0,
    }

    save_gene_pool(pool)
    print(f"Added '{name}' to gene pool (weight={new_weight:.4f}, {n_entries} total entries)")


def main():
    parser = argparse.ArgumentParser(description="Extract paper into gene pool")
    parser.add_argument("--name", required=True, help="Short name (e.g. 'mamba')")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="Path to local PDF file")
    source.add_argument("--arxiv", help="ArXiv paper ID (e.g. 2312.00752)")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set. Create evo/.env with GROQ_API_KEY=...")
        sys.exit(1)

    PAPERS_DIR.mkdir(exist_ok=True)

    # Get PDF
    if args.arxiv:
        pdf_path = download_arxiv(args.arxiv)
    else:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)
        dest = PAPERS_DIR / pdf_path.name
        if not dest.exists():
            shutil.copy2(pdf_path, dest)
        pdf_path = dest

    # Convert to markdown
    paper_md = convert_pdf_to_md(str(pdf_path))
    md_path = pdf_path.with_suffix(".md")
    md_path.write_text(paper_md)

    # Generate summary
    summary = generate_summary(paper_md)

    # Add to pool
    add_to_pool(args.name, summary, md_path)

    print(f"\nSummary:\n{summary}")


if __name__ == "__main__":
    main()
