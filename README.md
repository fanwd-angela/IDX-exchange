# IDX Exchange MLS Analytics Internship Project

## Project Overview

This repository contains Python scripts and generated output datasets completed during the IDX Exchange Data Analyst Internship Program.

The project analyzes real MLS listing and sold transaction data provided through the IDX Exchange pipeline. The workflow follows the internship curriculum:

- Data aggregation and preparation
- Dataset validation and exploratory analysis
- Data cleaning and transformation
- Feature engineering and market metric creation
- Outlier detection and data quality improvement
- Tableau dashboard preparation

The primary tools used are:
- Python (Pandas)
- CSV data processing
- Tableau Desktop

The MLS datasets are confidential business data and are used only for internship project purposes.

---

# Output CSV Files

Due to GitHub file size limitations, large CSV output files are stored externally. Download links for each week's generated outputs are provided below.

Each folder contains the CSV files generated from the corresponding week's Python scripts.

---

# Week 1 — Monthly Dataset Aggregation

## Objective

Combine monthly MLS listing and sold transaction files into unified datasets covering January 2024 through the latest available month.

## Processing Completed

- Loaded multiple monthly MLS Listing and Sold CSV files
- Concatenated monthly datasets
- Created combined listing and sold datasets
- Filtered datasets to Residential properties only
- Verified row counts before and after filtering

## Output Files

Download:

[Week 1 Output CSV Files](https://drive.google.com/drive/folders/1Y_5-H7usjcOgU0stJrtL2qeknzWA6P4-?usp=sharing)

---

# Weeks 2–3 — Dataset Validation, EDA, and Mortgage Rate Enrichment

## Objective

Validate dataset structure, analyze data quality, and enrich MLS transaction data with national 30-year fixed mortgage rates from FRED.

## Processing Completed

- Reviewed dataset structure and data types
- Generated missing value analysis
- Identified columns with high missing percentages
- Created numeric distribution summaries
- Added monthly mortgage rate data
- Validated successful data merges

## Output Files

Download:

[Weeks 2–3 Output CSV Files]
https://drive.google.com/drive/folders/1D5_yktSq2uN6KCplcTnPHtOL1TpWrKct?usp=sharing
https://drive.google.com/drive/folders/1Nra-kjmvbjFth1CuJecAdPrJzUxsLql4?usp=sharing

---

# Weeks 4–5 — Data Cleaning and Preparation

## Objective

Transform MLS datasets into clean, analysis-ready datasets.

## Processing Completed

- Converted date fields into datetime format
- Removed unnecessary columns
- Corrected numeric data types
- Handled missing values
- Flagged invalid numeric records
- Created date consistency validation flags:
  - listing_after_close_flag
  - purchase_after_close_flag
  - negative_timeline_flag
- Performed geographic data quality checks:
  - Missing coordinates
  - Invalid latitude and longitude values

## Output Files

Download:

[Weeks 4–5 Output CSV Files]
https://drive.google.com/drive/folders/106rFtw3BYN8X6aTsAWidyPJTB1EyqFc9?usp=sharing

---

# Week 6 — Feature Engineering and Market Metrics

## Objective

Create housing market indicators used for analytics and Tableau dashboards.

## Processing Completed

Created calculated housing market metrics including:

- Price Ratio
- Price Per Square Foot
- Days on Market
- Year / Month / YrMo
- Close-to-original-list ratio
- Listing-to-contract days
- Contract-to-close days

Generated segmented market summaries for analysis.

## Output Files

Download:

[Week 6 Output CSV Files](PASTE_LINK_HERE)

---

# Week 7 — Outlier Detection and Data Quality

## Objective

Identify extreme values while preserving original records.

## Processing Completed

- Applied IQR-based outlier detection
- Created outlier flag columns
- Compared dataset statistics before and after filtering
- Produced clean analysis datasets

## Output Files

Download:

[Week 7 Output CSV Files](PASTE_LINK_HERE)

---

# Weeks 8–10 — Tableau Dashboard Preparation

## Objective

Prepare cleaned and engineered datasets for Tableau dashboard development.

The datasets support:

### Market Analysis Dashboard

- Monthly median close price
- Average days on market
- Close-to-original-list price ratio
- New listings
- Closed sales
- Additional market analysis

### Competitive Analysis Dashboard

- Top listing agents
- Top listing offices
- Median close price analysis
- Homes sold analysis

## Output Files

Download:

[Tableau Dataset Files](PASTE_LINK_HERE)

---

# File Usage

After downloading the CSV files:

1. Extract the downloaded folder
2. Place the files in the project directory
3. Run the corresponding Python script

---

# Repository Structure
