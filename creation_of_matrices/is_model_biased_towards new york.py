import pandas as pd
import json
from collections import Counter

# Load hotel index-to-ID mapping
with open('hotel_idx_to_id.json') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

# Load hotel metadata CSV
hotels_df = pd.read_csv('hotel_df.csv')  # Replace with your actual filename

# Inspect columns to find the one with city/region
print("CSV Columns:", hotels_df.columns.tolist())

# Example: assume hotel ID is in 'hotel_id' and city is in 'city' column
# Filter only hotels used in the recommender
hotel_ids_in_model = set(idx_to_hotel_id.values())
filtered_hotels = hotels_df[hotels_df['offering_id'].isin(hotel_ids_in_model)]

# Count number of hotels per city
city_counts = Counter(filtered_hotels['locality'])

# Print results sorted by count
print("\nHotel count by city:")
for city, count in city_counts.most_common():
    print(f"{city}: {count}")


# Hotel count by city:
# New York City: 362
# Houston: 269
# San Antonio: 261
# Los Angeles: 220
# San Francisco: 212
# San Diego: 204
# Indianapolis: 145
# Dallas: 142
# Austin: 140
# Phoenix: 139
# Charlotte: 136
# Chicago: 130
# Washington DC: 114
# Columbus: 113
# Jacksonville: 112
# Denver: 106
# Seattle: 105
# Memphis: 96
# Fort Worth: 79
# Boston: 74
# Philadelphia: 72
# El Paso: 61
# Baltimore: 59
# San Jose: 52
# Detroit: 25