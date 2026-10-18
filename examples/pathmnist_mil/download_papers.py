"""Cache primary-source PDFs required by this example's research.yaml."""
from pathlib import Path

from src.research.config import LiteratureConfig
from src.research.literature import _extract_text, download

PAPERS = {
    "abmil_2018": "https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf",
    "attention2minority_2023": "https://arxiv.org/pdf/2301.07700",
    "psemix_2024": "https://arxiv.org/pdf/2306.16180",
}


def main():
    output = Path(__file__).resolve().parent / "papers"
    output.mkdir(exist_ok=True)
    for name, url in PAPERS.items():
        path = output / f"{name}.pdf"
        body = path.read_bytes() if path.exists() else download(url, LiteratureConfig(timeout_seconds=60))
        _extract_text(body, pdf=True)
        if not path.exists():
            path.write_bytes(body)
        print(f"Verified PDF: {name}", flush=True)


if __name__ == "__main__":
    main()
