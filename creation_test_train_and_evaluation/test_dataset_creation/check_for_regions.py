import pandas as pd

def get_and_print_unique_values(csv_path, col_name):
    try:
        df = pd.read_csv(csv_path)
        if col_name not in df.columns:
            print(f"Column '{col_name}' not found in the CSV.")
            return []
        unique_vals = df[col_name].dropna().unique().tolist()
        print(f"Unique values in column '{col_name}' ({len(unique_vals)} total):")
        for val in unique_vals:
            print(val)
        return unique_vals
    except Exception as e:
        print(f"Error reading CSV or processing column: {e}")
        return []

unique_locations = get_and_print_unique_values('hotel_df.csv', 'region')
print("Returned list:", unique_locations)