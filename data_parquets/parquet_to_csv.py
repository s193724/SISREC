import pandas as pd
import os
import pyarrow as pa

# List of Parquet file paths
parquet_files = [
    'hotel_df.parquet',
    'REVIEWS_DF.parquet',
    'USER_DF.parquet'
    # Add more file paths as needed
]

# Convert each Parquet file to CSV
for parquet_file in parquet_files:
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Generate CSV file name (replace .parquet with .csv)
    csv_file = os.path.splitext(parquet_file)[0] + '.csv'

    # Save DataFrame to CSV
    df.to_csv(csv_file, index=False)

    print(f"Converted {parquet_file} to {csv_file}")
