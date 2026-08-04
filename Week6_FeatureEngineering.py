"""Week 6: Feature engineering and market metrics for IDX Exchange.

This script starts from the Week 5 cleaned sold dataset and creates the market
analytics fields needed for Tableau dashboards and segmented analysis.

Transformations and rationale
-----------------------------
* price_ratio = ClosePrice / OriginalListPrice
  Measures negotiation strength against the original asking price.
* close_to_original_list_ratio = ClosePrice / OriginalListPrice
  Same business definition as price_ratio, retained with the explicit dashboard
  naming requested by the Week 6 assignment.
* price_per_sqft = ClosePrice / LivingArea
  Normalizes sale price by home size for cross-property comparison.
* days_on_market_metric = DaysOnMarket
  Preserves the raw time-to-sell field under a metric-friendly name.
* close_year, close_month, YrMo
  Derived from CloseDate for monthly time-series analysis.
* listing_to_contract_days = PurchaseContractDate - ListingContractDate
  Measures time from listing to accepted offer.
* contract_to_close_days = CloseDate - PurchaseContractDate
  Measures escrow / closing period duration.
* school_district_name
  Uses the best available district field already present in the MLS data. The
  handbook references California School District Areas 2024-25 for coordinate-
  based enrichment; boundary joins require a local geospatial boundary file and
  geospatial libraries, which are documented in the generated work report.

Outputs
-------
Week6_EngineeredMarketMetrics.csv
    Full Week 5 cleaned sold dataset with engineered Week 6 fields appended.
Week6_SampleOutput.csv
    Small proof table showing the engineered columns populated.
Week6_CountyOrParish_Summary.csv
    Segment summary grouped by CountyOrParish.
Week6_PropertyType_Summary.csv
    Segment summary grouped by PropertyType and PropertySubType.
Week6_OfficeCompetitiveSummary.csv
    Listing-office competitive summary for later Tableau analysis.
Week6_DataTypeConfirmations.csv
    Data type confirmation for engineered fields.
Week6_WorkReport.md
    Human-readable report summarizing transformations, counts, and outputs.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
]

NUMERIC_INPUT_COLUMNS = [
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
    "Latitude",
    "Longitude",
]

ENGINEERED_COLUMNS = [
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "days_on_market_metric",
    "close_year",
    "close_month",
    "YrMo",
    "listing_to_contract_days",
    "contract_to_close_days",
    "school_district_name",
    "school_district_source",
]

REQUIRED_COLUMNS = [
    *DATE_COLUMNS,
    *NUMERIC_INPUT_COLUMNS,
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor",
    "ListOfficeName",
    "BuyerOfficeName",
]


def require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only where denominator is positive; otherwise return NaN."""
    valid = denominator.notna() & denominator.gt(0)
    return numerator.where(valid).div(denominator.where(valid))


def first_non_null_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first available non-empty text value across candidate columns."""
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].replace("", pd.NA)
        result = result.fillna(values)
    return result


def add_engineered_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    for column in DATE_COLUMNS:
        data[column] = pd.to_datetime(data[column], errors="coerce")

    for column in NUMERIC_INPUT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["price_ratio"] = safe_ratio(data["ClosePrice"], data["OriginalListPrice"])
    data["close_to_original_list_ratio"] = data["price_ratio"]
    data["price_per_sqft"] = safe_ratio(data["ClosePrice"], data["LivingArea"])
    data["days_on_market_metric"] = data["DaysOnMarket"]

    data["close_year"] = data["CloseDate"].dt.year.astype("Int64")
    data["close_month"] = data["CloseDate"].dt.month.astype("Int64")
    data["YrMo"] = data["CloseDate"].dt.to_period("M").astype("string")

    data["listing_to_contract_days"] = (
        data["PurchaseContractDate"] - data["ListingContractDate"]
    ).dt.days.astype("Int64")
    data["contract_to_close_days"] = (
        data["CloseDate"] - data["PurchaseContractDate"]
    ).dt.days.astype("Int64")

    # Boundary-based enrichment can be added later from the CA school district
    # area layer. For the current deliverable, use the district fields already
    # present in the MLS extract so the output remains analysis-ready.
    district_candidates = [
        "HighSchoolDistrict",
        "ElementarySchoolDistrict",
        "MiddleOrJuniorSchoolDistrict",
    ]
    data["school_district_name"] = first_non_null_text(data, district_candidates)
    data["school_district_source"] = "not available in MLS fields"
    data.loc[
        data["school_district_name"].notna(),
        "school_district_source",
    ] = "MLS district field fallback"

    return data


def summarize_segment(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        data.groupby(group_columns, dropna=False)
        .agg(
            sold_records=("ListingKey", "count"),
            median_close_price=("ClosePrice", "median"),
            average_close_price=("ClosePrice", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            average_price_per_sqft=("price_per_sqft", "mean"),
            median_days_on_market=("days_on_market_metric", "median"),
            average_days_on_market=("days_on_market_metric", "mean"),
            median_close_to_original_list_ratio=(
                "close_to_original_list_ratio",
                "median",
            ),
            average_close_to_original_list_ratio=(
                "close_to_original_list_ratio",
                "mean",
            ),
            median_listing_to_contract_days=(
                "listing_to_contract_days",
                "median",
            ),
            median_contract_to_close_days=("contract_to_close_days", "median"),
        )
        .reset_index()
        .sort_values("sold_records", ascending=False)
    )
    return summary


def office_competitive_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("ListOfficeName", dropna=False)
        .agg(
            sold_records=("ListingKey", "count"),
            sales_volume=("ClosePrice", "sum"),
            median_close_price=("ClosePrice", "median"),
            median_days_on_market=("days_on_market_metric", "median"),
            median_close_to_original_list_ratio=(
                "close_to_original_list_ratio",
                "median",
            ),
            top_county=("CountyOrParish", lambda values: values.mode().iat[0] if not values.mode().empty else pd.NA),
        )
        .reset_index()
        .sort_values(["sales_volume", "sold_records"], ascending=[False, False])
    )


def data_type_confirmations(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": ENGINEERED_COLUMNS,
            "dtype": [str(data[column].dtype) for column in ENGINEERED_COLUMNS],
            "non_null_count": [int(data[column].notna().sum()) for column in ENGINEERED_COLUMNS],
            "missing_count": [int(data[column].isna().sum()) for column in ENGINEERED_COLUMNS],
        }
    )


def quality_counts(data: pd.DataFrame) -> dict[str, int]:
    counts = {
        "rows_with_price_ratio": int(data["price_ratio"].notna().sum()),
        "rows_missing_original_list_price_for_ratio": int(data["price_ratio"].isna().sum()),
        "rows_with_price_per_sqft": int(data["price_per_sqft"].notna().sum()),
        "rows_with_listing_to_contract_days": int(data["listing_to_contract_days"].notna().sum()),
        "rows_with_contract_to_close_days": int(data["contract_to_close_days"].notna().sum()),
        "rows_with_school_district_name": int(data["school_district_name"].notna().sum()),
        "rows_missing_school_district_name": int(data["school_district_name"].isna().sum()),
    }

    for column in [
        "missing_coordinates_flag",
        "zero_coordinate_flag",
        "positive_longitude_flag",
        "out_of_state_flag",
    ]:
        if column in data.columns:
            counts[column] = int(data[column].fillna(False).sum())

    return counts


def write_work_report(
    output_path: Path,
    input_path: Path,
    rows_before: int,
    rows_after: int,
    counts: dict[str, int],
    county_summary: pd.DataFrame,
    property_summary: pd.DataFrame,
) -> None:
    top_county = county_summary.iloc[0]
    top_property = property_summary.iloc[0]

    report = f"""# Week 6 Work Report - Feature Engineering and Market Metrics

## Assignment Requirement

Week 6 requires engineered market metrics for Tableau analysis: price ratio, close-to-original-list ratio, price per square foot, days on market, year/month/YrMo, listing-to-contract days, and contract-to-close days. It also requires a sample output table with the new fields populated and at least one segmented summary table grouped by PropertyType or CountyOrParish.

## Source Dataset

- Input file: `{input_path}`
- Rows before Week 6 feature engineering: {rows_before:,}
- Rows after Week 6 feature engineering: {rows_after:,}
- Row-level filtering performed in Week 6: none

## Transformations Completed

- Converted `CloseDate`, `PurchaseContractDate`, and `ListingContractDate` to datetime for reliable date arithmetic.
- Confirmed numeric types for `ClosePrice`, `OriginalListPrice`, `LivingArea`, `DaysOnMarket`, `Latitude`, and `Longitude`.
- Created `price_ratio` and `close_to_original_list_ratio` as `ClosePrice / OriginalListPrice`.
- Created `price_per_sqft` as `ClosePrice / LivingArea`.
- Created `days_on_market_metric` from the cleaned `DaysOnMarket` field.
- Created `close_year`, `close_month`, and `YrMo` from `CloseDate`.
- Created `listing_to_contract_days` from `PurchaseContractDate - ListingContractDate`.
- Created `contract_to_close_days` from `CloseDate - PurchaseContractDate`.
- Added `school_district_name` from available MLS district fields, preferring `HighSchoolDistrict`, then `ElementarySchoolDistrict`, then `MiddleOrJuniorSchoolDistrict`.

## Quality Checks

- Rows with populated price ratio: {counts["rows_with_price_ratio"]:,}
- Rows missing original list price for ratio: {counts["rows_missing_original_list_price_for_ratio"]:,}
- Rows with populated price per square foot: {counts["rows_with_price_per_sqft"]:,}
- Rows with listing-to-contract days: {counts["rows_with_listing_to_contract_days"]:,}
- Rows with contract-to-close days: {counts["rows_with_contract_to_close_days"]:,}
- Rows with school district name: {counts["rows_with_school_district_name"]:,}
- Rows missing school district name: {counts["rows_missing_school_district_name"]:,}

## Geographic Data Quality

- Missing coordinate records: {counts.get("missing_coordinates_flag", 0):,}
- Zero coordinate records: {counts.get("zero_coordinate_flag", 0):,}
- Positive longitude records: {counts.get("positive_longitude_flag", 0):,}
- Out-of-state or implausible coordinate records: {counts.get("out_of_state_flag", 0):,}

The Week 6 handbook references California School District Areas 2024-25 for latitude/longitude-based school district enrichment. The current local Python environment does not include geospatial join libraries such as geopandas or shapely, so the deliverable uses available MLS district fields and records the source in `school_district_source`. A later geospatial pass can replace or augment this field using the CA Open Data GeoJSON boundary layer.

## Segment Summary Highlights

- Largest county segment: {top_county["CountyOrParish"]} with {int(top_county["sold_records"]):,} sold records and median close price ${top_county["median_close_price"]:,.0f}.
- Largest property segment: {top_property["PropertyType"]} / {top_property["PropertySubType"]} with {int(top_property["sold_records"]):,} sold records and median close price ${top_property["median_close_price"]:,.0f}.

## Deliverable Files

- `Week6_FeatureEngineering.py`
- `Week6_EngineeredMarketMetrics.csv`
- `Week6_SampleOutput.csv`
- `Week6_CountyOrParish_Summary.csv`
- `Week6_PropertyType_Summary.csv`
- `Week6_OfficeCompetitiveSummary.csv`
- `Week6_DataTypeConfirmations.csv`
- `Week6_WorkReport.md`
"""
    output_path.write_text(report, encoding="utf-8")


def run_week6(input_path: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path, low_memory=False)
    require_columns(source)

    rows_before = len(source)
    engineered = add_engineered_metrics(source)
    rows_after = len(engineered)

    engineered_path = output_directory / "Week6_EngineeredMarketMetrics.csv"
    sample_path = output_directory / "Week6_SampleOutput.csv"
    county_summary_path = output_directory / "Week6_CountyOrParish_Summary.csv"
    property_summary_path = output_directory / "Week6_PropertyType_Summary.csv"
    office_summary_path = output_directory / "Week6_OfficeCompetitiveSummary.csv"
    dtypes_path = output_directory / "Week6_DataTypeConfirmations.csv"
    report_path = output_directory / "Week6_WorkReport.md"

    engineered.to_csv(engineered_path, index=False, date_format="%Y-%m-%d")

    sample_columns = [
        "ListingKey",
        "CloseDate",
        "ClosePrice",
        "OriginalListPrice",
        "LivingArea",
        "price_ratio",
        "close_to_original_list_ratio",
        "price_per_sqft",
        "days_on_market_metric",
        "YrMo",
        "listing_to_contract_days",
        "contract_to_close_days",
        "CountyOrParish",
        "PropertySubType",
        "school_district_name",
    ]
    engineered.loc[:, sample_columns].head(25).to_csv(sample_path, index=False)

    county_summary = summarize_segment(engineered, ["CountyOrParish"])
    county_summary.to_csv(county_summary_path, index=False)

    property_summary = summarize_segment(engineered, ["PropertyType", "PropertySubType"])
    property_summary.to_csv(property_summary_path, index=False)

    office_summary = office_competitive_summary(engineered)
    office_summary.to_csv(office_summary_path, index=False)

    dtype_summary = data_type_confirmations(engineered)
    dtype_summary.to_csv(dtypes_path, index=False)

    counts = quality_counts(engineered)
    write_work_report(
        report_path,
        input_path,
        rows_before,
        rows_after,
        counts,
        county_summary,
        property_summary,
    )

    print("WEEK 6 FEATURE ENGINEERING COMPLETE")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after: {rows_after:,}")
    print(f"Engineered dataset: {engineered_path}")
    print(f"Sample output: {sample_path}")
    print(f"County summary: {county_summary_path}")
    print(f"Property summary: {property_summary_path}")
    print(f"Office summary: {office_summary_path}")
    print(f"Data type confirmations: {dtypes_path}")
    print(f"Work report: {report_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Week 6 engineered market metrics and summary tables."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("Week5_CombinedSold_Cleaned.csv"),
        help="Clean Week 5 sold CSV (default: Week5_CombinedSold_Cleaned.csv)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="Directory for Week 6 deliverables (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_week6(args.input_csv, args.output_directory)


if __name__ == "__main__":
    main()
