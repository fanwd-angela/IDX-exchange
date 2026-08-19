"""Week 7: Flag IQR outliers and create an analysis-ready sold dataset.

The source file is never modified.  Every source row is retained in the
flagged output, with field-level flags for ClosePrice, LivingArea, and
DaysOnMarket.  The filtered output contains rows that pass all three IQR
tests.  Thresholds and before/after statistics are written separately so the
filtering decision is reproducible and easy to audit.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


FIELDS = ["ClosePrice", "LivingArea", "DaysOnMarket"]


def iqr_bounds(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": q1 - 1.5 * iqr,
        "upper_bound": q3 + 1.5 * iqr,
    }


def flag_outliers(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [field for field in FIELDS if field not in data]
    if missing:
        raise ValueError("Input is missing required fields: " + ", ".join(missing))

    threshold_rows = []
    field_flags = []
    for field in FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")
        bounds = iqr_bounds(data[field])
        flag_name = f"{field.lower()}_iqr_outlier_flag"
        data[flag_name] = (
            data[field].lt(bounds["lower_bound"])
            | data[field].gt(bounds["upper_bound"])
        )
        field_flags.append(flag_name)
        threshold_rows.append(
            {
                "field": field,
                **bounds,
                "non_null_count": int(data[field].notna().sum()),
                "outlier_count": int(data[flag_name].sum()),
                "outlier_percent": float(data[flag_name].mean() * 100),
            }
        )

    data["any_iqr_outlier_flag"] = data[field_flags].any(axis=1)
    data["iqr_outlier_field_count"] = data[field_flags].sum(axis=1).astype("int8")
    data["analysis_ready_flag"] = ~data["any_iqr_outlier_flag"]
    return data, pd.DataFrame(threshold_rows)


def comparison_table(flagged: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        before = pd.to_numeric(flagged[field], errors="coerce")
        after = pd.to_numeric(clean[field], errors="coerce")
        rows.append(
            {
                "field": field,
                "rows_before": len(flagged),
                "rows_after": len(clean),
                "rows_removed": len(flagged) - len(clean),
                "mean_before": before.mean(),
                "mean_after": after.mean(),
                "median_before": before.median(),
                "median_after": after.median(),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    source_path: Path,
    flagged: pd.DataFrame,
    clean: pd.DataFrame,
    thresholds: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Week 7 Work Report - Outlier Detection and Data Quality",
        "",
        "## Method",
        "",
        "IQR bounds were calculated independently for ClosePrice, LivingArea, "
        "and DaysOnMarket using Q1 - 1.5 x IQR and Q3 + 1.5 x IQR. The source "
        "and flagged datasets preserve all records; only the separate "
        "analysis-ready dataset excludes rows flagged in any required field.",
        "",
        "## Row Counts",
        "",
        f"- Source: `{source_path.name}`",
        f"- Rows before filtering: {len(flagged):,}",
        f"- Rows after filtering: {len(clean):,}",
        f"- Rows flagged/excluded: {len(flagged) - len(clean):,}",
        f"- Retention rate: {len(clean) / len(flagged):.2%}",
        "",
        "## IQR Thresholds",
        "",
        "| Field | Q1 | Q3 | Lower bound | Upper bound | Outliers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in thresholds.itertuples(index=False):
        lines.append(
            f"| {row.field} | {row.q1:,.2f} | {row.q3:,.2f} | "
            f"{row.lower_bound:,.2f} | {row.upper_bound:,.2f} | "
            f"{row.outlier_count:,} ({row.outlier_percent:.2f}%) |"
        )
    lines.extend(
        [
            "",
            "## Before and After Medians",
            "",
            "| Field | Before | After |",
            "|---|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.field} | {row.median_before:,.2f} | "
            f"{row.median_after:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Deliverables",
            "",
            "- `Week7_OutlierDetection.py`",
            "- `Week7_Residential_Flagged.csv`",
            "- `Week7_Residential_AnalysisReady.csv`",
            "- `Week7_IQR_Thresholds.csv`",
            "- `Week7_BeforeAfter_Comparison.csv`",
            "- `Week7_WorkReport.md`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Week6_EngineeredMarketMetrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    flagged, thresholds = flag_outliers(data)
    clean = flagged.loc[flagged["analysis_ready_flag"]].copy()
    comparison = comparison_table(flagged, clean)

    flagged.to_csv(args.output_dir / "Week7_Residential_Flagged.csv", index=False)
    clean.to_csv(args.output_dir / "Week7_Residential_AnalysisReady.csv", index=False)
    thresholds.to_csv(args.output_dir / "Week7_IQR_Thresholds.csv", index=False)
    comparison.to_csv(args.output_dir / "Week7_BeforeAfter_Comparison.csv", index=False)
    write_report(
        args.output_dir / "Week7_WorkReport.md",
        args.input,
        flagged,
        clean,
        thresholds,
        comparison,
    )

    print(f"Flagged rows: {len(flagged):,}")
    print(f"Analysis-ready rows: {len(clean):,}")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
