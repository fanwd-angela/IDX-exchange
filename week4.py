import pandas as pd

# Load dataset
sold = pd.read_csv(
    "CombinedSold_WithMortgageRates.csv",
    low_memory=False
)

print("Before cleaning")
print("Rows:", len(sold))
print("Columns:", len(sold.columns))

# Convert date columns to datetime
date_columns = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

for column in date_columns:
    if column in sold.columns:
        sold[column] = pd.to_datetime(
            sold[column],
            errors="coerce"
        )

print("\nDate conversion complete.")

# Convert important numeric columns
numeric_columns = [
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
    "rate_30yr_fixed"
]

for column in numeric_columns:
    if column in sold.columns:
        sold[column] = pd.to_numeric(
            sold[column],
            errors="coerce"
        )

print("Numeric conversion complete.")

# Missing value summary
missing_report = pd.DataFrame({
    "Missing Count": sold.isnull().sum(),
    "Missing Percent": sold.isnull().mean() * 100
})

missing_report.to_csv("Week4_MissingValueReport.csv")

print("\nMissing value report saved.")

# Remove unnecessary columns
drop_columns = [
    "MediaURL",
    "MediaURL2",
    "MediaURL3",
    "MediaURL4",
    "MediaURL5"
]

existing_drop_columns = [
    col
    for col in drop_columns
    if col in sold.columns
]

sold.drop(
    columns=existing_drop_columns,
    inplace=True
)

print("Dropped", len(existing_drop_columns), "unused columns.")

# Flag invalid numeric values
sold["invalid_closeprice_flag"] = (
    sold["ClosePrice"] <= 0
)

sold["invalid_livingarea_flag"] = (
    sold["LivingArea"] <= 0
)

sold["invalid_dom_flag"] = (
    sold["DaysOnMarket"] < 0
)

sold["invalid_bedrooms_flag"] = (
    sold["BedroomsTotal"] < 0
)

sold["invalid_bathrooms_flag"] = (
    sold["BathroomsTotalInteger"] < 0
)

print("\nInvalid value flags created.")

# Date consistency checks
sold["listing_after_close_flag"] = (
    sold["ListingContractDate"] >
    sold["CloseDate"]
)

sold["purchase_after_close_flag"] = (
    sold["PurchaseContractDate"] >
    sold["CloseDate"]
)

sold["negative_timeline_flag"] = (
    sold["PurchaseContractDate"] <
    sold["ListingContractDate"]
)

print("\nDate consistency flag counts")

print(
    "Listing after close:",
    sold["listing_after_close_flag"].sum()
)

print(
    "Purchase after close:",
    sold["purchase_after_close_flag"].sum()
)

print(
    "Negative timeline:",
    sold["negative_timeline_flag"].sum()
)

# Geographic quality checks
sold["missing_coordinates_flag"] = (
    sold["Latitude"].isna() |
    sold["Longitude"].isna()
)

sold["zero_coordinate_flag"] = (
    (sold["Latitude"] == 0) |
    (sold["Longitude"] == 0)
)

sold["positive_longitude_flag"] = (
    sold["Longitude"] > 0
)

sold["out_of_state_flag"] = (
    (sold["Latitude"] < 32) |
    (sold["Latitude"] > 43) |
    (sold["Longitude"] < -125) |
    (sold["Longitude"] > -114)
)

print("\nGeographic summary")

print(
    "Missing coordinates:",
    sold["missing_coordinates_flag"].sum()
)

print(
    "Zero coordinates:",
    sold["zero_coordinate_flag"].sum()
)

print(
    "Positive longitude:",
    sold["positive_longitude_flag"].sum()
)

print(
    "Out of state coordinates:",
    sold["out_of_state_flag"].sum()
)

# Remove records with impossible numeric values
rows_before = len(sold)

sold = sold[
    (sold["ClosePrice"] > 0) &
    (sold["LivingArea"] > 0) &
    (sold["DaysOnMarket"] >= 0) &
    (sold["BedroomsTotal"] >= 0) &
    (sold["BathroomsTotalInteger"] >= 0)
]

rows_after = len(sold)

print("\nRows before cleaning:", rows_before)
print("Rows after cleaning:", rows_after)
print("Rows removed:", rows_before - rows_after)

# Confirm data types
print("\nFinal data types")
print(sold.dtypes)

# Save cleaned dataset
sold.to_csv(
    "CombinedSold_Cleaned.csv",
    index=False
)

print("\nWeek 4-5 cleaning completed.")
print("Cleaned dataset saved as:")
print("CombinedSold_Cleaned.csv")