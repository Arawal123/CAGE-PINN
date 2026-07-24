from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_raw_records(directory: str | Path = "results/raw") -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_source_path"] = str(path)
        records.append(value)
    return records


def build_report(
    raw_directory: str | Path = "results/raw",
    processed_directory: str | Path = "results/processed",
    figure_directory: str | Path = "results/figures",
) -> dict[str, Any]:
    records = load_raw_records(raw_directory)
    processed = Path(processed_directory)
    figures = Path(figure_directory)
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    ledger_rows = []
    for record in records:
        config = record["config"]
        rows.append(
            {
                "run_id": record["run_id"],
                "status": record["status"],
                "problem": config["problem"],
                "backbone": config["backbone"],
                "method": config["method"],
                "seed": config["seed"],
                "steps": record.get("optimizer_steps", 0),
                "relative_l2": record.get("metrics", {}).get(
                    "relative_l2_post_training"
                ),
                "wall_seconds": record.get("total_seconds"),
                "ad_tokens": record.get("ledger", {}).get("spent_tokens"),
                "source": record["_source_path"],
            }
        )
        for category, tokens in record.get("ledger", {}).get(
            "totals_by_category", {}
        ).items():
            ledger_rows.append(
                {
                    "run_id": record["run_id"],
                    "method": config["method"],
                    "category": category,
                    "tokens": tokens,
                }
            )
    frame = pd.DataFrame(rows)
    table_path = processed / "runs.csv"
    frame.to_csv(table_path, index=False)
    ledger_path = processed / "compute_ledger.csv"
    pd.DataFrame(ledger_rows).to_csv(ledger_path, index=False)
    if not frame.empty:
        failure_table = (
            frame.assign(failed=frame["status"] != "completed")
            .groupby(["problem", "backbone", "method"], dropna=False)["failed"]
            .agg(["sum", "count"])
            .reset_index()
        )
        failure_table["failure_rate"] = failure_table["sum"] / failure_table["count"]
    else:
        failure_table = pd.DataFrame(
            columns=["problem", "backbone", "method", "sum", "count", "failure_rate"]
        )
    failure_path = processed / "failure_rates.csv"
    failure_table.to_csv(failure_path, index=False)
    figure_paths: list[str] = []
    if not frame.empty and frame["relative_l2"].notna().any():
        plot_frame = frame.dropna(subset=["relative_l2", "ad_tokens"])
        figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for method, group in plot_frame.groupby("method"):
            axis.scatter(group["ad_tokens"], group["relative_l2"], label=method, alpha=0.8)
        axis.set_xlabel("Charged symbolic AD tokens")
        axis.set_ylabel("Post-training relative $L^2$ error")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.2)
        axis.legend()
        axis.set_title("Smoke diagnostics — not publication evidence")
        path = figures / "smoke_error_vs_tokens.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(str(path))
    for record in records:
        history = record.get("history", [])
        if not history:
            continue
        steps = [entry["step"] for entry in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        axes[0].plot(
            steps,
            [entry["train"]["aggregate"] for entry in history],
            label="learner",
        )
        axes[0].plot(
            steps,
            [entry["monitor"]["aggregate"] for entry in history],
            label="monitor",
        )
        axes[0].set_xlabel("Optimizer step")
        axes[0].set_ylabel("Bounded residual-risk aggregate")
        axes[0].set_title("Residual generalization diagnostic")
        axes[0].legend()
        axes[0].grid(True, alpha=0.2)
        for atom in ("S", "J", "W"):
            axes[1].plot(
                steps,
                [
                    entry["allocation_requested"].get(atom, 0.0)
                    for entry in history
                ],
                label=atom,
            )
        axes[1].set_xlabel("Optimizer step")
        axes[1].set_ylabel("Requested compute share")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Controller allocation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)
        path = figures / f"{record['run_id']}_audit_controller.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(str(path))
    summary = {
        "raw_records": len(records),
        "completed": int(sum(record["status"] == "completed" for record in records)),
        "failed": int(sum(record["status"] != "completed" for record in records)),
        "table": str(table_path),
        "compute_ledger_table": str(ledger_path),
        "failure_rate_table": str(failure_path),
        "figures": figure_paths,
        "warning": "Generated only from immutable raw records; smoke data cannot support method claims.",
    }
    (processed / "report_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
