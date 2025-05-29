import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import json

# Step 1: Load hotel metadata and reviews
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')

# Step 2: Load existing user mappings
with open('user_id_to_idx.json', 'r') as f:
    user_id_to_idx = json.load(f)

with open('idx_to_user_id.json', 'r') as f:
    idx_to_user_id = json.load(f)
    idx_to_user_id = {int(k): v for k, v in idx_to_user_id.items()}

# Order users according to existing mapping index (to maintain consistent order)
ordered_user_ids = [idx_to_user_id[i] for i in range(len(idx_to_user_id))]

# Step 3: Merge hotel class info into reviews
df_reviews = df_reviews.merge(
    df_hotels[["offering_id", "hotel_class"]],
    on="offering_id",
    how="left"
)

# Fill missing hotel_class with median hotel class
median_hotel_class = df_hotels['hotel_class'].median()
df_reviews['hotel_class'] = df_reviews['hotel_class'].fillna(median_hotel_class)

# Step 4: Compute weighted score per review
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Step 5: Filter reviews to users present in the mapping only
df_filtered = df_reviews[df_reviews['id_user'].isin(ordered_user_ids)]

# Step 6: Create user-item matrix (pivot) with weighted scores
# Reindex rows by ordered_user_ids to keep consistent user order and fill missing with zeros
user_item_matrix = df_filtered.pivot_table(
    index='id_user',
    columns='offering_id',
    values='weighted_score',
    fill_value=0
).reindex(ordered_user_ids, axis=0, fill_value=0)

# Step 7: Convert the user-item matrix to a sparse matrix
sparse_user_features = csr_matrix(user_item_matrix.values)

# Step 8: Save sparse user features matrix to disk
save_npz('user_features_collab.npz', sparse_user_features)

print(f"User features matrix shape (users x hotels): {sparse_user_features.shape}")
print(f"Number of users: {len(ordered_user_ids)}")
print(f"Number of users in matrix: {user_item_matrix.shape[0]}")

#(62006, 3428)