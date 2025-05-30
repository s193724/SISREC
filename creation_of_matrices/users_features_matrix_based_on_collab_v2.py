import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import json

# Load hotel metadata & reviews
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')

# Load user mappings
with open('user_id_to_idx.json', 'r') as f:
    user_id_to_idx = json.load(f)

with open('idx_to_user_id.json', 'r') as f:
    idx_to_user_id = json.load(f)
    idx_to_user_id = {int(k): v for k, v in idx_to_user_id.items()}

# Order users by mapping to keep consistent ordering
ordered_user_ids = [idx_to_user_id[i] for i in range(len(idx_to_user_id))]

# Merge hotel class info into reviews
df_reviews = df_reviews.merge(
    df_hotels[['offering_id', 'hotel_class']],
    on='offering_id',
    how='left'
)

# Fill missing hotel_class with median
median_class = df_hotels['hotel_class'].median()
df_reviews['hotel_class'] = df_reviews['hotel_class'].fillna(median_class)

# Compute weighted score per review (mean of review columns * hotel_class)
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Filter reviews only for users in mapping
df_filtered = df_reviews[df_reviews['id_user'].isin(ordered_user_ids)]

# Create user-item pivot table (user x hotel)
user_item_matrix = df_filtered.pivot_table(
    index='id_user',
    columns='offering_id',
    values='weighted_score',
    fill_value=0
).reindex(ordered_user_ids, axis=0, fill_value=0)

# Convert to sparse matrix
user_features_collab = csr_matrix(user_item_matrix.values)

# Save sparse matrix for future use
save_npz('user_features_collab.npz', user_features_collab)

print(f"User features matrix shape (users x hotels): {user_features_collab.shape}")
print(f"Number of users: {len(ordered_user_ids)}")
print(f"Number of hotels: {user_features_collab.shape[1]}")

# Aggregate hotel features by taking average of review columns
hotel_features_df = df_reviews.groupby('offering_id')[review_cols].mean().reset_index()

# Ensure hotel order consistent with columns in user_item_matrix
hotel_ids_ordered = user_item_matrix.columns.to_list()

# Reindex hotel_features_df by hotel_ids_ordered to align
hotel_features_df = hotel_features_df.set_index('offering_id').reindex(hotel_ids_ordered, fill_value=0).reset_index()

# Convert to numpy array


hotel_features_sparse = csr_matrix(hotel_features_df)

save_npz('hotel_features.npz', hotel_features_sparse)
print(f"Hotel features matrix shape: {hotel_features_sparse.shape}")  # (num_hotels, 7)
