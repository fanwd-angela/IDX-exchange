import pandas as pd
import glob
import os

# Find all monthly Listing and Sold CSV files
listing_files = sorted(glob.glob("csv/CRMLSListing*.csv"))
sold_files = sorted(glob.glob("csv/CRMLSSold*.csv"))

print("Listing files found:", len(listing_files))
print("Sold files found:", len(sold_files))

# Store each monthly Listing DataFrame
listing_dfs = []
# Read every monthly Listing CSV and print its row count
for file in listing_files:
    df = pd.read_csv(file)
    print(f"{os.path.basename(file)} rows before concat: {len(df)}")
    listing_dfs.append(df)

# Store each monthly Sold DataFrame
sold_dfs = []
#read every monthly Sold CSV and print its row count
for file in sold_files:
    df = pd.read_csv(file)
    print(f"{os.path.basename(file)} rows before concat: {len(df)}")
    sold_dfs.append(df)

# Combine all monthly Listing datasets into one DataFrame
combined_listing = pd.concat(listing_dfs, ignore_index=True)
# Combine all monthly Sold datasets into one DataFrame
combined_sold = pd.concat(sold_dfs, ignore_index=True)

print("Listing rows after concat:", len(combined_listing))
print("Sold rows after concat:", len(combined_sold))

print("Listing rows before Residential filter:", len(combined_listing))
print("Sold rows before Residential filter:", len(combined_sold))

# Keep only Residential listings
combined_listing = combined_listing[
    combined_listing["PropertyType"] == "Residential"
]
# Keep only Residential sold properties
combined_sold = combined_sold[
    combined_sold["PropertyType"] == "Residential"
]

# Print row counts after filtering
print("Listing rows after Residential filter:", len(combined_listing))
print("Sold rows after Residential filter:", len(combined_sold))

# Save the combined Residential datasets as new CSV files
combined_listing.to_csv("CombinedListings.csv", index=False)
combined_sold.to_csv("CombinedSold.csv", index=False)

print("Done. Files saved.")