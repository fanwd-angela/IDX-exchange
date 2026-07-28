"""Weeks 4–5: Prepare MLS sold data for reliable analysis.

Outputs
-------
Week5_CombinedSold_Cleaned.csv
    Analysis-ready records with data-quality flags.
Week5_MissingValueReport_Before.csv
    Missing-value counts and percentages before cleaning.
Week5_MissingValueReport_After.csv
    Missing-value counts and percentages after cleaning.
Week5_CleaningSummary.csv
    Before/after counts and quality-check totals for auditing.

Missing-value policy
--------------------
* Rows missing fields required for core price/size/market-time analysis are
  removed: ClosePrice, LivingArea, DaysOnMarket, BedroomsTotal, and
  BathroomsTotalInteger.
* Missing dates are retained because they can still support non-timeline
  analyses. Missing dates are identified by dedicated flags.
* Missing coordinates are retained and flagged because these records can still
  support non-geographic analyses.
* Missing optional descriptive fields are retained rather than guessed or
  imputed. Inventing MLS attributes could bias later analysis.
"""

from pathlib import Path
import argparse

import pandas as pd


DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_COLUMNS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "Latitude",
    "Longitude",
    "YearBuilt",
    "rate_30yr_fixed",
]

# Large media URLs are not needed for statistical analysis.
DROP_COLUMNS = [
    "MediaURL",
    "MediaURL2",
    "MediaURL3",
    "MediaURL4",
    "MediaURL5",
]

# These fields are necessary for the intended core housing analysis.
REQUIRED_ANALYSIS_COLUMNS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
]

REQUIRED_INPUT_COLUMNS = [
    *DATE_COLUMNS,
    *REQUIRED_ANALYSIS_COLUMNS,
    "Latitude",
    "Longitude",
]


def missing_value_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an auditable missing-value summary for every column."""
    report = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().to_numpy(),
            "missing_percent": (frame.isna().mean() * 100).to_numpy(),
        }
    )
    return report.sort_values(
        ["missing_percent", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def require_columns(frame: pd.DataFrame) -> None:
    """Fail clearly if the source does not contain required MLS fields."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Input file is missing required columns: " + ", ".join(missing)
        )


def count_true(frame: pd.DataFrame, column: str) -> int:
    """Count True values in a boolean quality-control column."""
    return int(frame[column].fillna(False).sum())


def clean_mls_data(
    input_path: Path,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the MLS file, save deliverables, and return data plus summary."""
    sold = pd.read_csv(input_path, low_memory=False)
    require_columns(sold)

    rows_before = len(sold)
    columns_before = len(sold.columns)

    before_report = missing_value_report(sold)
    before_report.to_csv(
        output_directory / "Week5_MissingValueReport_Before.csv",
        index=False,
    )

    # Invalid text becomes NaT so malformed dates are handled consistently.
    for column in DATE_COLUMNS:
        sold[column] = pd.to_datetime(sold[column], errors="coerce")

    # Invalid text becomes NaN so it can be counted and handled explicitly.
    for column in NUMERIC_COLUMNS:
        if column in sold.columns:
            sold[column] = pd.to_numeric(sold[column], errors="coerce")

    # Record missing critical values before removing affected records.
    sold["missing_required_numeric_flag"] = sold[
        REQUIRED_ANALYSIS_COLUMNS
    ].isna().any(axis=1)

    # Flag impossible numeric values before filtering them out.
    sold["invalid_closeprice_flag"] = sold["ClosePrice"].le(0)
    sold["invalid_livingarea_flag"] = sold["LivingArea"].le(0)
    sold["invalid_dom_flag"] = sold["DaysOnMarket"].lt(0)
    sold["invalid_bedrooms_flag"] = sold["BedroomsTotal"].lt(0)
    sold["invalid_bathrooms_flag"] = sold[
        "BathroomsTotalInteger"
    ].lt(0)

    # Retain incomplete dates, but identify records that cannot be fully
    # validated. Comparisons involving NaT correctly evaluate to False.
    sold["missing_timeline_date_flag"] = sold[
        ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
    ].isna().any(axis=1)

    # Required logical order: listing <= purchase <= close.
    sold["listing_after_close_flag"] = (
        sold["ListingContractDate"] > sold["CloseDate"]
    )
    sold["purchase_after_close_flag"] = (
        sold["PurchaseContractDate"] > sold["CloseDate"]
    )
    sold["negative_timeline_flag"] = (
        sold["PurchaseContractDate"] < sold["ListingContractDate"]
    )

    # California coordinates are approximately within this bounding box.
    # Coordinate issues are flagged, not removed, so non-spatial analysis
    # remains possible.
    sold["missing_coordinates_flag"] = (
        sold["Latitude"].isna() | sold["Longitude"].isna()
    )
    sold["zero_coordinate_flag"] = (
        sold["Latitude"].eq(0) | sold["Longitude"].eq(0)
    )
    sold["positive_longitude_flag"] = sold["Longitude"].gt(0)
    sold["out_of_state_flag"] = (
        sold["Latitude"].lt(32)
        | sold["Latitude"].gt(43)
        | sold["Longitude"].lt(-125)
        | sold["Longitude"].gt(-114)
    )

    # Capture all requested audit totals before invalid records are removed.
    audit_columns = [
        "missing_required_numeric_flag",
        "invalid_closeprice_flag",
        "invalid_livingarea_flag",
        "invalid_dom_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
        "missing_timeline_date_flag",
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag",
        "missing_coordinates_flag",
        "zero_coordinate_flag",
        "positive_longitude_flag",
        "out_of_state_flag",
    ]
    audit_counts = {
        f"{column}_count_before_filtering": count_true(sold, column)
        for column in audit_columns
    }

    invalid_numeric_mask = (
        sold["missing_required_numeric_flag"]
        | sold["invalid_closeprice_flag"]
        | sold["invalid_livingarea_flag"]
        | sold["invalid_dom_flag"]
        | sold["invalid_bedrooms_flag"]
        | sold["invalid_bathrooms_flag"]
    )
    sold = sold.loc[~invalid_numeric_mask].copy()

    # Remove exact duplicate records after type normalization.
    duplicate_rows_removed = int(sold.duplicated().sum())
    sold = sold.drop_duplicates().copy()

    existing_drop_columns = [
        column for column in DROP_COLUMNS if column in sold.columns
    ]
    sold = sold.drop(columns=existing_drop_columns)

    rows_after = len(sold)
    cleaned_path = output_directory / "Week5_CombinedSold_Cleaned.csv"
    # Keep true datetime dtypes in memory for validation, while writing dates
    # to the CSV in a portable ISO format.
    sold.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d")

    after_report = missing_value_report(sold)
    after_report.to_csv(
        output_directory / "Week5_MissingValueReport_After.csv",
        index=False,
    )

    summary_values = {
        "input_file": str(input_path),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed_total": rows_before - rows_after,
        "duplicate_rows_removed": duplicate_rows_removed,
        "columns_before": columns_before,
        "columns_after": len(sold.columns),
        "unused_columns_removed": len(existing_drop_columns),
        **audit_counts,
    }
    summary = pd.DataFrame(
        {
            "metric": summary_values.keys(),
            "value": summary_values.values(),
        }
    )
    summary.to_csv(
        output_directory / "Week5_CleaningSummary.csv",
        index=False,
    )

    print("\nWEEK 5 MLS CLEANING SUMMARY")
    print(summary.to_string(index=False))
    print("\nFINAL DATA TYPES")
    print(sold.dtypes.to_string())
    print(f"\nCleaned dataset saved to: {cleaned_path}")

    return sold, summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and validate the Weeks 4–5 MLS sold dataset."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("CombinedSold_WithMortgageRates.csv"),
        help="Source CSV (default: CombinedSold_WithMortgageRates.csv)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="Directory for Week 5 deliverables (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    clean_mls_data(args.input_csv, args.output_directory)


if __name__ == "__main__":
    main()
