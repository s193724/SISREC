import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import json

# Step 1: Load hotel metadata and reviews
df_hotels = pd.read_csv('hotel_df.csv')
df = pd.read_csv('REVIEWS_DF.csv')

# Step 2: Load existing user mappings
with open('user_id_to_idx.json', 'r') as f:
    user_id_to_idx = json.load(f)

with open('idx_to_user_id.json', 'r') as f:
    idx_to_user_id = json.load(f)
    idx_to_user_id = {int(k): v for k, v in idx_to_user_id.items()}

# Order users according to existing mapping
ordered_user_ids = [idx_to_user_id[i] for i in range(len(idx_to_user_id))]

# Step 3: Merge hotel class info
df = df.merge(
    df_hotels[["offering_id", "hotel_class"]],
    on="offering_id",
    how="left"
)

# Fill missing hotel_class with median
median_hotel_class = df_hotels['hotel_class'].median()
df['hotel_class'] = df['hotel_class'].fillna(median_hotel_class)

# Step 4: Compute weighted score per review
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df['weighted_score'] = df[review_cols].mean(axis=1) * df['hotel_class']

# Step 5: Filter reviews to only users present in existing mapping
df_filtered = df[df['id_user'].isin(ordered_user_ids)]

# Step 6: Create user-item matrix pivot, reindexed by ordered_user_ids (to align with mapping)
user_item_matrix = df_filtered.pivot_table(
    index='id_user',
    columns='offering_id',
    values='weighted_score',
    fill_value=0
).reindex(ordered_user_ids, axis=0, fill_value=0)

# Step 7: Convert to sparse matrix
sparse_user_item = csr_matrix(user_item_matrix.values)

# Step 8: Compute cosine similarity (sparse)
similarity_matrix = cosine_similarity(sparse_user_item, dense_output=False)

# Step 9: Save sparse similarity matrix
save_npz("user_similarity_collab.npz", similarity_matrix)

print(f"Shape of user similarity matrix: {similarity_matrix.shape}")
print(f"Number of users in mapping: {len(user_id_to_idx)}")
print(f"Number of users in user-item matrix: {user_item_matrix.shape[0]}")
