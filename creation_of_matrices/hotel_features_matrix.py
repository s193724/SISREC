import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load hotel and review data
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')

# Fill missing hotel_class with median
median_hotel_class = df_hotels['hotel_class'].median()
df_hotels['hotel_class'] = df_hotels['hotel_class'].fillna(median_hotel_class)

# Merge hotel_class to reviews
df_reviews = df_reviews.merge(
    df_hotels[['offering_id', 'hotel_class', 'region']],
    on='offering_id',
    how='left'
)

# Review columns to average
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']

# Compute weighted score for each review (mean review score * hotel_class)
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Aggregate weighted score per hotel (mean across all reviews)
hotel_weighted_scores = df_reviews.groupby('offering_id')['weighted_score'].mean().reset_index()

# Merge back hotel region info for one-hot encoding
hotel_features = hotel_weighted_scores.merge(
    df_hotels[['offering_id', 'region']],
    on='offering_id',
    how='left'
)

# Fill missing region with 'Unknown'
hotel_features['region'] = hotel_features['region'].fillna('Unknown')

# One-hot encode regions
ohe = OneHotEncoder(sparse=True)
region_ohe = ohe.fit_transform(hotel_features[['region']])

# Convert weighted scores to sparse matrix (shape: num_hotels x 1)
weighted_score_sparse = csr_matrix(hotel_features['weighted_score'].values).T

# Combine weighted scores + one-hot regions horizontally
hotel_features_sparse = hstack([weighted_score_sparse, region_ohe])

# Save matrix and encoder for future use
save_npz('hotel_features.npz', hotel_features_sparse)

import joblib
joblib.dump(ohe, 'hotel_region_ohe.pkl')

print(f"Hotel features matrix shape: {hotel_features_sparse.shape}")
print(f"Columns: 1 (weighted score) + {region_ohe.shape[1]} (one-hot regions)")
joblib.dump(ohe, 'hotel_region_ohe.pkl')