import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import json

# --- Load mappings ---
with open("user_id_to_idx.json", 'r') as f:
    user_id_to_idx = json.load(f)

with open("idx_to_user_id.json", 'r') as f:
    idx_to_user_id = {int(k): v for k, v in json.load(f).items()}

with open("hotel_id_to_idx.json", 'r') as f:
    hotel_id_to_idx = json.load(f)

with open("hotel_idx_to_id.json", 'r') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

# --- Load data ---
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')

# --- Convert IDs to string to ensure consistent types ---
df_reviews['id_user'] = df_reviews['id_user'].astype(str)
df_reviews['offering_id'] = df_reviews['offering_id'].astype(str)
df_hotels['offering_id'] = df_hotels['offering_id'].astype(str)

print(f"Original reviews: {len(df_reviews)}")

# --- Merge hotel class info ---
df_reviews = df_reviews.merge(
    df_hotels[['offering_id', 'hotel_class']],
    on='offering_id',
    how='left'
)

# --- Fill missing hotel_class with median ---
median_hotel_class = df_hotels['hotel_class'].median()
df_reviews['hotel_class'] = df_reviews['hotel_class'].fillna(median_hotel_class)

# --- Compute weighted score ---
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# --- Filter reviews to only users and hotels in mappings ---
df_filtered = df_reviews[
    df_reviews['id_user'].isin(user_id_to_idx) &
    df_reviews['offering_id'].isin(hotel_id_to_idx)
]

print(f"Filtered reviews: {len(df_filtered)}")
print("Non-zero weighted scores:", (df_filtered['weighted_score'] != 0).sum())

# --- Matrix creation ---
num_users = len(user_id_to_idx)
num_hotels = len(hotel_id_to_idx)

rows = []
cols = []
data = []

for _, row in df_filtered.iterrows():
    user_idx = user_id_to_idx[row['id_user']]
    hotel_idx = hotel_id_to_idx[row['offering_id']]
    score = row['weighted_score']
    rows.append(user_idx)
    cols.append(hotel_idx)
    data.append(score)

user_item_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_hotels))

# --- Save matrix ---
save_npz("user_hotel_matrix.npz", user_item_matrix)

print(f"User-item matrix shape: {user_item_matrix.shape}")
print(f"Number of users: {num_users}, number of hotels: {num_hotels}")
print(f"Non-zero elements in matrix: {user_item_matrix.nnz}")
print(f"Sparsity: {100 * (1 - user_item_matrix.nnz / (num_users * num_hotels)):.2f}%")

# --- Optional: Verify saved matrix ---
loaded = load_npz("user_hotel_matrix.npz")
assert (loaded != user_item_matrix).nnz == 0
print("Matrix saved and loaded successfully.")
