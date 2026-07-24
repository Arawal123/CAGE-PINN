from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="research/claim_evidence/ledger.csv")
    parser.add_argument("--paper", default="paper/main.tex")
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    ledger = pd.read_csv(args.ledger).fillna("")
    major = ledger[ledger["claim_id"].isin(["C002", "C003", "C004", "C005"])]
    unresolved = major[major["status"] != "supported"]
    paper = Path(args.paper).read_text(encoding="utf-8")
    placeholders = paper.count("RESULTS_REQUIRED") + paper.count(r"RESULTS\_REQUIRED")
    payload = {
        "unresolved_major_claims": unresolved["claim_id"].tolist(),
        "result_placeholders": placeholders,
        "publication_ready": unresolved.empty and placeholders == 0,
    }
    print(payload)
    if not payload["publication_ready"] and not args.allow_draft:
        raise SystemExit(
            "Paper build blocked: unresolved major claims or numerical placeholders remain"
        )


if __name__ == "__main__":
    main()
