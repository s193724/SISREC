import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack, save_npz
import json

# Load user profile data_and_RS
df_users = pd.read_csv('USER_DF.csv')

# Example numeric columns - adjust according to your data_and_RS
numeric_cols = ['num_helpful_votes_user', 'num_cities', 'num_reviews_profile']

# Handle missing numeric values by filling median
for col in numeric_cols:
    median_val = df_users[col].median()
    df_users[col] = df_users[col].fillna(median_val)

# Scale numeric features
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df_users[numeric_cols])

# Example text column: user location - can add more text fields similarly
vectorizer_location = TfidfVectorizer(max_features=500)
location_features = vectorizer_location.fit_transform(df_users['location_user'].astype(str))

# If you have other text or categorical features, encode them similarly:
# For example, if you had 'user_type' categorical:
# vectorizer_type = TfidfVectorizer(max_features=50)
# type_features = vectorizer_type.fit_transform(df_users['user_type'].astype(str))

# Combine all features horizontally (numeric + text)
user_features_sparse = hstack([csr_matrix(numeric_features), location_features]).tocsr()

# Save user_id to index mappings
user_ids = df_users['id_user'].tolist()
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}

with open('user_id_to_idx.json', 'w') as f:
    json.dump(user_id_to_idx, f)
with open('idx_to_user_id.json', 'w') as f:
    json.dump(idx_to_user_id, f)

# Save the sparse user features matrix
save_npz('user_features_sparse.npz', user_features_sparse)

print(f"User features matrix shape: {user_features_sparse.shape}")
