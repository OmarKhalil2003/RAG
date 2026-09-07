import sys
import json
from pathlib import Path

# Ensure src is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.ingestion.parser import BilingualPDFParser
from legal_rag.ingestion.validator import CorpusValidator


def main():
    print("=" * 60)
    print("Starting Legal RAG Multi-Statute Corpus Ingestion")
    print("=" * 60)

    registry_path = project_root / "data" / "corpora_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            corpora = json.load(f)
    else:
        config_path = project_root / "data" / "corpus_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            corpora = [json.load(f)]

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for corpus in corpora:
        print("\n" + "=" * 60)
        print(f"Ingesting: {corpus.get('law_name_en')} ({corpus.get('law_name_ar')})")
        print(f"Law Type: {corpus.get('law_type')} | Year: {corpus.get('law_year')}")
        print(f"Expected range: {corpus.get('expected_article_range')}")
        print("=" * 60)

        pdf_rel = corpus.get("source_pdf", "")
        pdf_path = project_root / pdf_rel
        if not pdf_path.exists():
            # Fallback to root or law_books
            pdf_path = project_root / Path(pdf_rel).name
        if not pdf_path.exists():
            pdf_path = project_root / "data" / "law_books" / Path(pdf_rel).name

        print(f"Resolved PDF Path: {pdf_path}")
        if not pdf_path.exists():
            print(f"Error: PDF not found at {pdf_path}")
            continue

        parser = BilingualPDFParser(corpus)
        articles = parser.parse(pdf_path)

        expected_range = corpus.get("expected_article_range", [1, len(articles)])
        validator = CorpusValidator(expected_range=expected_range)
        report = validator.validate(articles)

        print("\n" + "-" * 50)
        print(f"VALIDATION REPORT [{corpus.get('id')}]:")
        print(report.summary)
        if report.missing_articles:
            print(f"Missing articles ({len(report.missing_articles)}): {report.missing_articles[:20]}")
        if report.repealed_articles:
            print(f"Repealed provisions ({len(report.repealed_articles)}): {report.repealed_articles[:10]} ...")
        print("-" * 50)

        processed_rel = corpus.get("processed_file", f"data/processed/{corpus.get('id')}.json")
        output_file = project_root / processed_rel
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in articles], f, ensure_ascii=False, indent=2)

        print(f"Successfully serialized {len(articles)} articles to: {output_file}")

    print("\n" + "=" * 60)
    print("Status: ALL CORPORA INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
