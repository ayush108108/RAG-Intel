"""Fetch optional load-test fixtures from their original sources at run time.

The arXiv PDFs are not bundled or covered by this repository's MIT licence.
The versions linked below use arXiv's non-exclusive distribution licence:
https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html
That licence grants distribution rights to arXiv, not to this repository.
Review each paper's licence before reuse or redistribution; terms can vary
between versions: https://info.arxiv.org/help/license/index.html

Other downloads retain their source terms, including Project Gutenberg's
geographic restrictions and notices: https://www.gutenberg.org/policy/license.html
Download-only PDFs and the original Sherlock Holmes text stay gitignored.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "load_test"

URLS = {
    # Academic papers: pinned versions, with licence information on the abs pages.
    # Attention Is All You Need: https://arxiv.org/abs/1706.03762v7
    "attention_paper.pdf": "https://arxiv.org/pdf/1706.03762v7",
    # Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980v9
    "adam_optimizer.pdf": "https://arxiv.org/pdf/1412.6980v9",
    # Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385v1
    "resnet_paper.pdf": "https://arxiv.org/pdf/1512.03385v1",
    # Python documentation.
    "python_json_docs.html": "https://docs.python.org/3/library/json.html",
    "python_sqlite_docs.html": "https://docs.python.org/3/library/sqlite3.html",
    "python_asyncio_docs.html": "https://docs.python.org/3/library/asyncio.html",
    # Public CSV and JSON datasets.
    "airport_codes.csv": (
        "https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv"
    ),
    "gdp_data.csv": "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv",
    "github_emojis.json": "https://api.github.com/emojis",
    "country_codes.json": (
        "https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.json"
    ),
    # Project Gutenberg books; preserve the original notices in local downloads.
    "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
    "sherlock_holmes.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
    # Images with text for optional OCR.
    "ocr_test_1.png": "https://tesseract.projectnaptha.com/img/eng_bw.png",
    "ocr_test_2.png": "https://raw.githubusercontent.com/tesseract-ocr/tessdoc/main/eurotext.png",
}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading optional fixtures to sample_data/load_test...")
    print("arXiv PDFs retain their source licence; review the module's licence notes.")
    failed = 0
    for filename, url in URLS.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"[Skipping] {filename} already exists.")
            continue
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RAG-Intel/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(dest)
            print(f"[Done] {filename}")
        except Exception:
            partial.unlink(missing_ok=True)
            failed += 1
            print(f"[Failed] {filename}; retry or retrieve it from its source URL.")
    print(f"Download complete: {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
