import pandas as pd
from pathlib import Path

# File locations

# Folder containing this week3.py file
project_folder = Path(__file__).resolve().parent

# Week 1 combined datasets
sold_file = project_folder / "CombinedSold.csv"
listings_file = project_folder / "CombinedListings.csv"

# Local backup of the downloaded FRED mortgage-rate CSV
mortgage_backup_file = project_folder / "csv" / "MORTGAGE30US.csv"

# Output files
sold_output_file = project_folder / "CombinedSold_WithMortgageRates.csv"
listings_output_file = project_folder / "CombinedListings_WithMortgageRates.csv"
mortgage_monthly_output_file = project_folder / "MortgageRates_Monthly.csv"


# Step 1: Load the combined MLS datasets

print("Loading combined MLS datasets...")

sold = pd.read_csv(sold_file, low_memory=False)
listings = pd.read_csv(listings_file, low_memory=False)

print("Sold rows loaded:", len(sold))
print("Listing rows loaded:", len(listings))


# Step 2: Fetch weekly 30-year mortgage rates from FRED

fred_url = (
    "https://fred.stlouisfed.org/graph/"
    "fredgraph.csv?id=MORTGAGE30US"
)

try:
    print("\nAttempting to download mortgage-rate data from FRED...")

    mortgage = pd.read_csv(
        fred_url,
        parse_dates=["observation_date"]
    )

    print("FRED data downloaded successfully.")

except Exception as error:
    print("\nFRED download failed.")
    print("Reason:", error)
    print("Using the local mortgage-rate CSV instead.")

    mortgage = pd.read_csv(
        mortgage_backup_file,
        parse_dates=["observation_date"]
    )


# Step 3: Validate and rename mortgage-rate columns

required_mortgage_columns = [
    "observation_date",
    "MORTGAGE30US"
]

missing_mortgage_columns = [
    column
    for column in required_mortgage_columns
    if column not in mortgage.columns
]

if missing_mortgage_columns:
    raise KeyError(
        "The mortgage-rate CSV is missing these columns: "
        f"{missing_mortgage_columns}"
    )

mortgage = mortgage.rename(
    columns={
        "observation_date": "date",
        "MORTGAGE30US": "rate_30yr_fixed"
    }
)

# Ensure mortgage rates are numeric
mortgage["rate_30yr_fixed"] = pd.to_numeric(
    mortgage["rate_30yr_fixed"],
    errors="coerce"
)

# Remove rows with invalid dates or missing mortgage rates
mortgage = mortgage.dropna(
    subset=["date", "rate_30yr_fixed"]
)

print("\nWeekly mortgage-rate rows loaded:", len(mortgage))
print(mortgage.head())


# Step 4: Resample weekly rates to monthly averages

mortgage["year_month"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage.groupby("year_month", as_index=False)["rate_30yr_fixed"]
    .mean()
)

# Round the monthly average rate for cleaner output
mortgage_monthly["rate_30yr_fixed"] = (
    mortgage_monthly["rate_30yr_fixed"].round(3)
)

print("\nMonthly mortgage-rate preview:")
print(mortgage_monthly.tail(12))

# Save the monthly mortgage-rate table
mortgage_monthly.to_csv(
    mortgage_monthly_output_file,
    index=False
)



# Step 5: Validate required MLS date columns

if "CloseDate" not in sold.columns:
    raise KeyError(
        "CombinedSold.csv does not contain a CloseDate column."
    )

if "ListingContractDate" not in listings.columns:
    raise KeyError(
        "CombinedListings.csv does not contain a "
        "ListingContractDate column."
    )



# Step 6: Create year-month keys in the MLS datasets

# Sold records are matched using CloseDate
sold["CloseDate"] = pd.to_datetime(
    sold["CloseDate"],
    errors="coerce"
)

sold["year_month"] = sold["CloseDate"].dt.to_period("M")

# Listing records are matched using ListingContractDate
listings["ListingContractDate"] = pd.to_datetime(
    listings["ListingContractDate"],
    errors="coerce"
)

listings["year_month"] = (
    listings["ListingContractDate"].dt.to_period("M")
)

print("\nSold rows with invalid or missing CloseDate:")
print(sold["CloseDate"].isna().sum())

print("\nListing rows with invalid or missing ListingContractDate:")
print(listings["ListingContractDate"].isna().sum())


# Step 7: Merge monthly mortgage rates onto both datasets

sold_with_rates = sold.merge(
    mortgage_monthly,
    on="year_month",
    how="left",
    validate="many_to_one"
)

listings_with_rates = listings.merge(
    mortgage_monthly,
    on="year_month",
    how="left",
    validate="many_to_one"
)



# Step 8: Validate row counts after the merge

print("\nRow-count validation:")

print("Sold rows before merge:", len(sold))
print("Sold rows after merge:", len(sold_with_rates))

print("Listing rows before merge:", len(listings))
print("Listing rows after merge:", len(listings_with_rates))

if len(sold) != len(sold_with_rates):
    raise ValueError(
        "Sold row count changed during the merge."
    )

if len(listings) != len(listings_with_rates):
    raise ValueError(
        "Listing row count changed during the merge."
    )



# Step 9: Check for unmatched mortgage-rate values

sold_null_rates = (
    sold_with_rates["rate_30yr_fixed"].isna().sum()
)

listing_null_rates = (
    listings_with_rates["rate_30yr_fixed"].isna().sum()
)

print("\nMortgage-rate null validation:")
print("Sold rows with null mortgage rate:", sold_null_rates)
print(
    "Listing rows with null mortgage rate:",
    listing_null_rates
)

# Show which months did not match, if any
if sold_null_rates > 0:
    sold_unmatched_months = (
        sold_with_rates.loc[
            sold_with_rates["rate_30yr_fixed"].isna(),
            "year_month"
        ]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nUnmatched sold year-month values:")
    print(sold_unmatched_months)

if listing_null_rates > 0:
    listing_unmatched_months = (
        listings_with_rates.loc[
            listings_with_rates["rate_30yr_fixed"].isna(),
            "year_month"
        ]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nUnmatched listing year-month values:")
    print(listing_unmatched_months)

if sold_null_rates == 0 and listing_null_rates == 0:
    print(
        "\nValidation passed: no null mortgage-rate "
        "values exist after the merge."
    )
else:
    print(
        "\nValidation warning: some rows have no matching "
        "mortgage rate. Review missing or invalid MLS dates "
        "and unmatched year-month values shown above."
    )



# Step 10: Preview the enriched datasets

sold_preview_columns = [
    column
    for column in [
        "CloseDate",
        "year_month",
        "ClosePrice",
        "rate_30yr_fixed"
    ]
    if column in sold_with_rates.columns
]

listing_preview_columns = [
    column
    for column in [
        "ListingContractDate",
        "year_month",
        "ListPrice",
        "rate_30yr_fixed"
    ]
    if column in listings_with_rates.columns
]

print("\nEnriched sold dataset preview:")
print(sold_with_rates[sold_preview_columns].head())

print("\nEnriched listings dataset preview:")
print(listings_with_rates[listing_preview_columns].head())


# Step 11: Save both enriched datasets


sold_with_rates.to_csv(
    sold_output_file,
    index=False
)

listings_with_rates.to_csv(
    listings_output_file,
    index=False
)

print("\nWeek 3 mortgage enrichment completed.")
print("Files saved:")
print(sold_output_file.name)
print(listings_output_file.name)
print(mortgage_monthly_output_file.name)