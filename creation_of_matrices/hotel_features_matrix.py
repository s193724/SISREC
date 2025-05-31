import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

# Load hotel and review data
df_hotels = pd.read_csv('hotel_df.csv')
df_reviews = pd.read_csv('REVIEWS_DF.csv')


# Fill missing hotel_class with median
median_hotel_class = df_hotels['hotel_class'].median()
df_hotels['hotel_class'] = df_hotels['hotel_class'].fillna(median_hotel_class)

# Merge hotel_class and region to reviews
df_reviews = df_reviews.merge(
    df_hotels[['offering_id', 'hotel_class', 'region']],
    on='offering_id',
    how='left'
)

# Review columns
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']

# Compute weighted score
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Compute per-hotel average of each category
category_means = df_reviews.groupby('offering_id')[review_cols].mean().reset_index()

# Compute per-hotel weighted score
hotel_weighted_scores = df_reviews.groupby('offering_id')['weighted_score'].mean().reset_index()

# Merge everything
hotel_features = df_hotels[['offering_id', 'region', 'hotel_class']].merge(
    category_means, on='offering_id', how='left'
).merge(
    hotel_weighted_scores, on='offering_id', how='left'
)

hotel_features[review_cols] = hotel_features[review_cols].fillna(hotel_features[review_cols].median())
hotel_features['weighted_score'] = hotel_features['weighted_score'].fillna(hotel_features['weighted_score'].median())
# Fill missing region
hotel_features['region'] = hotel_features['region'].fillna('Unknown')
hotel_features['hotel_class'] = hotel_features['hotel_class'].fillna(hotel_features['hotel_class'].median())

# One-hot encode region
ohe = OneHotEncoder(sparse=True)
region_ohe = ohe.fit_transform(hotel_features[['region']])

# Build sparse feature matrix
review_feature_sparse = csr_matrix(hotel_features[review_cols].values)
weighted_score_sparse = csr_matrix(hotel_features[['weighted_score']].values)
hotel_class_sparse = csr_matrix(hotel_features[['hotel_class']].values)
hotel_features_sparse = hstack([review_feature_sparse, weighted_score_sparse, region_ohe, hotel_class_sparse])


print("Any NaNs in hotel_features_sparse.data?", np.isnan(hotel_features_sparse.data).any())
# Save outputs
save_npz('hotel_features.npz', hotel_features_sparse)
joblib.dump(ohe, 'hotel_region_ohe.pkl')

print(f"Hotel features matrix shape: {hotel_features_sparse.shape}")
print("Columns: ", review_cols + ['weighted_score'] + list(ohe.get_feature_names_out(['region'])) + ['hotel_class'])