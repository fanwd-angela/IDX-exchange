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
  Derived from each property's Latitude and Longitude using the California
  School District Areas 2024-25 boundary file.

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
Week6_SchoolDistrict_Summary.csv
    School-district segment summary from boundary-enriched fields.
Week6_DataTypeConfirmations.csv
    Data type confirmation for engineered fields.
Week6_WorkReport.md
    Human-readable report summarizing transformations, counts, and outputs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

try:
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
except ImportError:  # pragma: no cover - reported clearly at runtime.
    Point = None
    STRtree = None
    shape = None


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
    "school_district_type",
    "school_district_county",
    "school_district_cdcode",
    "school_district_match_count",
    "elementary_school_district_boundary",
    "high_school_district_boundary",
    "unified_school_district_boundary",
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

SCHOOL_DISTRICT_BOUNDARY_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)


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


def load_school_district_boundaries(boundary_path: Path) -> list[dict[str, object]]:
    """Load CA school district polygons and required attributes."""
    if not boundary_path.exists():
        raise FileNotFoundError(
            "School district boundary file not found: "
            f"{boundary_path}. Download it from {SCHOOL_DISTRICT_BOUNDARY_URL}"
        )
    if Point is None or STRtree is None or shape is None:
        raise ImportError(
            "shapely is required for the school district boundary join. "
            "Install it with: python -m pip install shapely"
        )

    with boundary_path.open(encoding="utf-8") as boundary_file:
        geojson = json.load(boundary_file)

    boundaries = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        boundaries.append(
            {
                "geometry": shape(feature.get("geometry")),
                "DistrictName": properties.get("DistrictName"),
                "DistrictType": properties.get("DistrictType"),
                "CountyName": properties.get("CountyName"),
                "CDCode": properties.get("CDCode"),
            }
        )
    return boundaries


def choose_primary_district(matches: list[dict[str, object]]) -> dict[str, object] | None:
    """Prefer Unified districts, then Elementary, then High for one display field."""
    if not matches:
        return None

    priority = {"Unified": 0, "Elementary": 1, "High": 2}
    return sorted(
        matches,
        key=lambda item: (
            priority.get(str(item.get("DistrictType")), 99),
            str(item.get("DistrictName")),
        ),
    )[0]


def add_school_district_boundaries(
    data: pd.DataFrame,
    boundary_path: Path,
) -> pd.DataFrame:
    """Spatially enrich records with CA school district boundary attributes."""
    boundaries = load_school_district_boundaries(boundary_path)
    geometries = [boundary["geometry"] for boundary in boundaries]
    tree = STRtree(geometries)

    valid_coordinate_mask = (
        data["Latitude"].notna()
        & data["Longitude"].notna()
        & data["Latitude"].between(32, 43)
        & data["Longitude"].between(-125, -114)
    )
    unique_coordinates = (
        data.loc[valid_coordinate_mask, ["Latitude", "Longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows = []
    for row in unique_coordinates.itertuples(index=False):
        latitude = float(row.Latitude)
        longitude = float(row.Longitude)
        point = Point(longitude, latitude)
        candidate_indexes = tree.query(point)
        matches = [
            boundaries[int(index)]
            for index in candidate_indexes
            if geometries[int(index)].covers(point)
        ]
        primary = choose_primary_district(matches)
        by_type = {
            str(match.get("DistrictType")): match.get("DistrictName")
            for match in matches
            if match.get("DistrictType") and match.get("DistrictName")
        }
        rows.append(
            {
                "Latitude": latitude,
                "Longitude": longitude,
                "school_district_name": (
                    primary.get("DistrictName") if primary else pd.NA
                ),
                "school_district_type": (
                    primary.get("DistrictType") if primary else pd.NA
                ),
                "school_district_county": (
                    primary.get("CountyName") if primary else pd.NA
                ),
                "school_district_cdcode": (
                    primary.get("CDCode") if primary else pd.NA
                ),
                "school_district_match_count": len(matches),
                "elementary_school_district_boundary": by_type.get(
                    "Elementary",
                    pd.NA,
                ),
                "high_school_district_boundary": by_type.get("High", pd.NA),
                "unified_school_district_boundary": by_type.get("Unified", pd.NA),
            }
        )

    match_frame = pd.DataFrame(rows)
    if match_frame.empty:
        data["school_district_match_count"] = 0
        data["school_district_source"] = "no valid coordinates for boundary join"
        return data

    data = data.merge(match_frame, on=["Latitude", "Longitude"], how="left")
    data["school_district_match_count"] = (
        data["school_district_match_count"].fillna(0).astype("Int64")
    )
    data["school_district_source"] = "CA School District Areas 2024-25 boundary join"
    data.loc[
        data["school_district_match_count"].eq(0),
        "school_district_source",
    ] = "no boundary match from valid Latitude/Longitude"
    data.loc[
        data["Latitude"].isna() | data["Longitude"].isna(),
        "school_district_source",
    ] = "missing Latitude/Longitude"

    fallback = first_non_null_text(
        data,
        [
            "HighSchoolDistrict",
            "ElementarySchoolDistrict",
            "MiddleOrJuniorSchoolDistrict",
        ],
    )
    data["school_district_name"] = data["school_district_name"].fillna(fallback)
    data.loc[
        data["school_district_match_count"].eq(0) & fallback.notna(),
        "school_district_source",
    ] = "MLS district field fallback after no boundary match"

    return data


def add_engineered_metrics(frame: pd.DataFrame, boundary_path: Path) -> pd.DataFrame:
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

    data = add_school_district_boundaries(data, boundary_path)

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


def school_district_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(
            [
                "school_district_name",
                "school_district_type",
                "school_district_county",
            ],
            dropna=False,
        )
        .agg(
            sold_records=("ListingKey", "count"),
            median_close_price=("ClosePrice", "median"),
            average_close_price=("ClosePrice", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_days_on_market=("days_on_market_metric", "median"),
            median_close_to_original_list_ratio=(
                "close_to_original_list_ratio",
                "median",
            ),
        )
        .reset_index()
        .sort_values("sold_records", ascending=False)
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
        "rows_with_boundary_school_district_match": int(
            data["school_district_match_count"].fillna(0).gt(0).sum()
        ),
        "rows_with_multiple_boundary_district_matches": int(
            data["school_district_match_count"].fillna(0).gt(1).sum()
        ),
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
    school_summary: pd.DataFrame,
) -> None:
    top_county = county_summary.iloc[0]
    top_property = property_summary.iloc[0]
    top_school = school_summary.dropna(subset=["school_district_name"]).iloc[0]

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
- Added school district fields from the California School District Areas 2024-25 boundary file using each property's `Latitude` and `Longitude`.
- Created `elementary_school_district_boundary`, `high_school_district_boundary`, and `unified_school_district_boundary` because the state layer contains overlapping district types.
- Created `school_district_name` as the primary boundary match, preferring Unified, then Elementary, then High district records.

## Quality Checks

- Rows with populated price ratio: {counts["rows_with_price_ratio"]:,}
- Rows missing original list price for ratio: {counts["rows_missing_original_list_price_for_ratio"]:,}
- Rows with populated price per square foot: {counts["rows_with_price_per_sqft"]:,}
- Rows with listing-to-contract days: {counts["rows_with_listing_to_contract_days"]:,}
- Rows with contract-to-close days: {counts["rows_with_contract_to_close_days"]:,}
- Rows with school district name: {counts["rows_with_school_district_name"]:,}
- Rows missing school district name: {counts["rows_missing_school_district_name"]:,}
- Rows with boundary-derived school district match: {counts["rows_with_boundary_school_district_match"]:,}
- Rows with multiple boundary district matches: {counts["rows_with_multiple_boundary_district_matches"]:,}

## Geographic Data Quality

- Missing coordinate records: {counts.get("missing_coordinates_flag", 0):,}
- Zero coordinate records: {counts.get("zero_coordinate_flag", 0):,}
- Positive longitude records: {counts.get("positive_longitude_flag", 0):,}
- Out-of-state or implausible coordinate records: {counts.get("out_of_state_flag", 0):,}

School district enrichment used the California School District Areas 2024-25 GeoJSON boundary layer from CA Open Data:

`{SCHOOL_DISTRICT_BOUNDARY_URL}`

## Segment Summary Highlights

- Largest county segment: {top_county["CountyOrParish"]} with {int(top_county["sold_records"]):,} sold records and median close price ${top_county["median_close_price"]:,.0f}.
- Largest property segment: {top_property["PropertyType"]} / {top_property["PropertySubType"]} with {int(top_property["sold_records"]):,} sold records and median close price ${top_property["median_close_price"]:,.0f}.
- Largest school district segment: {top_school["school_district_name"]} with {int(top_school["sold_records"]):,} sold records and median close price ${top_school["median_close_price"]:,.0f}.

## Deliverable Files

- `Week6_FeatureEngineering.py`
- `Week6_EngineeredMarketMetrics.csv`
- `Week6_SampleOutput.csv`
- `Week6_CountyOrParish_Summary.csv`
- `Week6_PropertyType_Summary.csv`
- `Week6_OfficeCompetitiveSummary.csv`
- `Week6_SchoolDistrict_Summary.csv`
- `Week6_DataTypeConfirmations.csv`
- `Week6_WorkReport.md`
"""
    output_path.write_text(report, encoding="utf-8")


def run_week6(input_path: Path, output_directory: Path, boundary_path: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path, low_memory=False)
    require_columns(source)

    rows_before = len(source)
    engineered = add_engineered_metrics(source, boundary_path)
    rows_after = len(engineered)

    engineered_path = output_directory / "Week6_EngineeredMarketMetrics.csv"
    sample_path = output_directory / "Week6_SampleOutput.csv"
    county_summary_path = output_directory / "Week6_CountyOrParish_Summary.csv"
    property_summary_path = output_directory / "Week6_PropertyType_Summary.csv"
    office_summary_path = output_directory / "Week6_OfficeCompetitiveSummary.csv"
    school_summary_path = output_directory / "Week6_SchoolDistrict_Summary.csv"
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
        "school_district_type",
        "elementary_school_district_boundary",
        "high_school_district_boundary",
        "unified_school_district_boundary",
        "school_district_source",
    ]
    engineered.loc[:, sample_columns].head(25).to_csv(sample_path, index=False)

    county_summary = summarize_segment(engineered, ["CountyOrParish"])
    county_summary.to_csv(county_summary_path, index=False)

    property_summary = summarize_segment(engineered, ["PropertyType", "PropertySubType"])
    property_summary.to_csv(property_summary_path, index=False)

    office_summary = office_competitive_summary(engineered)
    office_summary.to_csv(office_summary_path, index=False)

    school_summary = school_district_summary(engineered)
    school_summary.to_csv(school_summary_path, index=False)

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
        school_summary,
    )

    print("WEEK 6 FEATURE ENGINEERING COMPLETE")
    print(f"Rows before: {rows_before:,}")
    print(f"Rows after: {rows_after:,}")
    print(f"Engineered dataset: {engineered_path}")
    print(f"Sample output: {sample_path}")
    print(f"County summary: {county_summary_path}")
    print(f"Property summary: {property_summary_path}")
    print(f"Office summary: {office_summary_path}")
    print(f"School district summary: {school_summary_path}")
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
    parser.add_argument(
        "--school-district-boundaries",
        type=Path,
        default=Path("ca_school_district_areas_2024_25.geojson"),
        help=(
            "California School District Areas 2024-25 GeoJSON boundary file "
            "(default: ca_school_district_areas_2024_25.geojson)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_week6(
        args.input_csv,
        args.output_directory,
        args.school_district_boundaries,
    )


if __name__ == "__main__":
    main()
