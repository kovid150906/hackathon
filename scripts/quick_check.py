import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import NarrativeConsistencyChecker


def main() -> None:
    df = pd.read_csv("train.csv")
    row = df.iloc[0].to_dict()

    book = row["book_name"]
    backstory = row["content"]

    with open(f"data/{book}.txt", "r", encoding="utf-8", errors="ignore") as fh:
        narrative = fh.read()

    checker = NarrativeConsistencyChecker("config.yaml")
    res = checker.process_single_example(
        narrative,
        backstory,
        story_id=str(row["id"]),
        narrative_id=book,
    )

    print("decision=", res["decision"], "confidence=", res["confidence"])
    print("evidence_count=", res["evidence_count"])


if __name__ == "__main__":
    main()
