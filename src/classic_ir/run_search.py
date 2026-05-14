from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classic_ir.search import ClassicSearchEngine


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "artifacts" / "classic_ir"


def print_results(label: str, results) -> None:
    print(f"\n{label}")
    if not results:
        print("No results.")
        return
    for result in results:
        print(f"{result.rank}. score={result.score:.4f} id={result.doc_id}")
        print(f"   title: {result.title}")
        print(f"   snippet: {result.snippet[:300]}")


def main() -> None:
    engine = ClassicSearchEngine.load(str(INDEX_DIR))

    sample_queries = [
        "quyền sử dụng đất",
        "hợp đồng lao động",
    ]
    for query in sample_queries:
        print_results(f"BM25: {query}", engine.bm25_search(query, top_k=5))

    print_results("Boolean: đất đai AND thừa kế", engine.boolean_search("đất đai AND thừa kế", top_k=5))
    print_results('Phrase: "quyền sử dụng đất"', engine.phrase_search('"quyền sử dụng đất"', top_k=5))
    print_results(
        'Phrase + BM25 rank: "hợp đồng lao động"',
        engine.phrase_search_ranked('"hợp đồng lao động"', top_k=5),
    )
    print_results("Proximity: đất NEAR/5 thừa kế", engine.near_search("đất", "thừa kế", distance=5, top_k=5))
    print_results("Proximity: hợp đồng NEAR/3 lao động", engine.near_search("hợp đồng", "lao động", distance=3, top_k=5))


if __name__ == "__main__":
    main()
