import pandas as pd
import numpy as np
import faiss
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz

# Step 1: Load Recommendation_System_Logic_Code
df_profiles = pd.read_csv("USER_DF.csv")

numeric_cols = ['num_helpful_votes_user', 'num_cities', 'num_reviews_profile']

# Step 2: Impute missing numeric values with median
for col in numeric_cols:
    median_val = df_profiles[col].median()
    df_profiles[col] = df_profiles[col].fillna(median_val)

# Step 3: Fill missing location_user with a placeholder string
df_profiles['location_user'] = df_profiles['location_user'].fillna('unknown')

# Step 4: Now select the columns without dropping rows
df = df_profiles[['id_user'] + numeric_cols + ['location_user']]

# Step 5: Prepare numeric features
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[numeric_cols])

# Step 6: TF-IDF for location
vectorizer = TfidfVectorizer(max_features=500)
location_features = vectorizer.fit_transform(df['location_user'].astype(str))

# Step 7: Combine features and convert to dense
from scipy.sparse import hstack
combined_sparse = hstack([numeric_features, location_features])
combined_dense = combined_sparse.toarray().astype('float32')

# Step 8: Normalize for cosine similarity (unit vectors)
norms = np.linalg.norm(combined_dense, axis=1, keepdims=True)
combined_dense = combined_dense / np.clip(norms, 1e-8, None)

# Step 9: Faiss index (cosine similarity via inner product)
index = faiss.IndexFlatIP(combined_dense.shape[1])
index.add(combined_dense)

# Step 10: Search for top-K neighbors
k = 50
similarities, indices = index.search(combined_dense, k)

# Step 11: Build sparse similarity matrix
rows, cols, data = [], [], []
num_users = len(df)
user_ids = df['id_user'].tolist()
id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
idx_to_id = {idx: uid for uid, idx in id_to_idx.items()}

for i in range(num_users):
    for j in range(k):
        neighbor_idx = indices[i][j]
        sim = similarities[i][j]
        if i != neighbor_idx:
            rows.append(i)
            cols.append(neighbor_idx)
            data.append(sim)

sparse_sim_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_users))
save_npz("metadata_user_user.npz", sparse_sim_matrix)

# Save mappings
with open("user_id_to_idx.json", "w") as f:
    json.dump(id_to_idx, f)
with open("idx_to_user_id.json", "w") as f:
    json.dump(idx_to_id, f)

print(f"Sparse similarity matrix saved: shape={sparse_sim_matrix.shape}")

#shape=(62006, 62006)