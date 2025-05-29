import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load hotel metadata and reviews
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')

# Load mappings
with open('hotel_id_to_idx.json') as f:
    hotel_id_to_idx = json.load(f)

with open('user_id_to_idx.json') as f:
    user_id_to_idx = json.load(f)

# Fill missing hotel_class
median_class = df_hotels['hotel_class'].median()
df_hotels['hotel_class'] = df_hotels['hotel_class'].fillna(median_class)

# Encode categorical hotel features (region, type, locality) with TF-IDF
tfidf_region = TfidfVectorizer(max_features=50)
region_features = tfidf_region.fit_transform(df_hotels['region'].astype(str))

tfidf_type = TfidfVectorizer(max_features=20)
type_features = tfidf_type.fit_transform(df_hotels['type'].astype(str))

tfidf_locality = TfidfVectorizer(max_features=100)
locality_features = tfidf_locality.fit_transform(df_hotels['locality'].astype(str))

# Numeric hotel_class scaled
scaler = StandardScaler()
hotel_class_scaled = scaler.fit_transform(df_hotels[['hotel_class']])

# Combine hotel metadata features horizontally
from scipy.sparse import csr_matrix, hstack

hotel_meta_features = hstack([
    csr_matrix(hotel_class_scaled),
    region_features,
    type_features,
    locality_features
])

# Filter reviews to known hotels and users
df_reviews = df_reviews[
    df_reviews['offering_id'].isin(hotel_id_to_idx.keys()) &
    df_reviews['id_user'].isin(user_id_to_idx.keys())
]

# Compute weighted score
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df_reviews = df_reviews.merge(df_hotels[['offering_id', 'hotel_class']], on='offering_id', how='left')
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Prepare sparse hotel-user matrix
rows, cols, data = [], [], []
for _, row in df_reviews.iterrows():
    hotel_idx = hotel_id_to_idx[row['offering_id']]
    user_idx = user_id_to_idx[row['id_user']]
    rows.append(hotel_idx)
    cols.append(user_idx)
    data.append(row['weighted_score'])

num_hotels = len(hotel_id_to_idx)
num_users = len(user_id_to_idx)
hotel_user_matrix = csr_matrix((data, (rows, cols)), shape=(num_hotels, num_users))

# Combine user rating matrix with hotel metadata features
# hotel_user_matrix: (num_hotels x num_users)
# hotel_meta_features: (num_hotels x meta_feature_dim)
combined_hotel_features = hstack([hotel_user_matrix, hotel_meta_features])

# Compute cosine similarity between hotels using combined features
hotel_similarity = cosine_similarity(combined_hotel_features, dense_output=False)

# Save the similarity matrix
save_npz('hotel_similarity_matrix.npz', hotel_similarity)

print(f"Hotel similarity matrix with metadata shape: {hotel_similarity.shape}")
#Hotel similarity matrix with metadata shape: (3428, 3428)