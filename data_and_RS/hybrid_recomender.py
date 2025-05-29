import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
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

# Convert IDs to string for consistency
df_reviews['id_user'] = df_reviews['id_user'].astype(str)
df_reviews['offering_id'] = df_reviews['offering_id'].astype(str)
df_hotels['offering_id'] = df_hotels['offering_id'].astype(str)

# Merge hotel class info (fill missing with median)
median_hotel_class = df_hotels['hotel_class'].median()
df_reviews = df_reviews.merge(df_hotels[['offering_id', 'hotel_class']], on='offering_id', how='left')
df_reviews['hotel_class'] = df_reviews['hotel_class'].fillna(median_hotel_class)

# Compute weighted score (mean review * hotel_class)
review_cols = ['service', 'cleanliness', 'overall', 'value', 'location', 'sleep_quality', 'rooms']
df_reviews['weighted_score'] = df_reviews[review_cols].mean(axis=1) * df_reviews['hotel_class']

# Filter for known users and hotels
df_filtered = df_reviews[df_reviews['id_user'].isin(user_id_to_idx) & df_reviews['offering_id'].isin(hotel_id_to_idx)]

# Build sparse user-item matrix
rows, cols, data = [], [], []
for _, row in df_filtered.iterrows():
    rows.append(user_id_to_idx[row['id_user']])
    cols.append(hotel_id_to_idx[row['offering_id']])
    data.append(row['weighted_score'])

num_users = len(user_id_to_idx)
num_hotels = len(hotel_id_to_idx)
user_item_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_hotels))

# Save user-item matrix
save_npz("user_hotel_matrix.npz", user_item_matrix)

# Compute user-user similarity matrix (cosine)
user_similarity = cosine_similarity(user_item_matrix, dense_output=False)
save_npz("user_similarity_collab.npz", user_similarity)

# Load hotel similarity matrix (precomputed)
hotel_similarity = load_npz("hotel_similarity_matrix.npz")

# --- Collaborative Filtering Recommender ---
def recommend_collab(user_id, top_n=10):
    if user_id not in user_id_to_idx:
        print("User not found!")
        return []

    user_idx = user_id_to_idx[user_id]
    sim_scores = user_similarity[user_idx].toarray().flatten()

    # Find top similar users except self
    sim_users_idx = np.argsort(sim_scores)[::-1]
    sim_users_idx = sim_users_idx[sim_users_idx != user_idx]

    # Aggregate ratings from similar users
    sim_users = sim_users_idx[:top_n]
    weights = sim_scores[sim_users]

    # Weighted sum of other users' ratings
    weighted_ratings = weights @ user_item_matrix[sim_users]
    recommended_idx = np.argsort(weighted_ratings)[::-1]

    # Filter out items already rated by user
    user_rated = user_item_matrix[user_idx].toarray().flatten() > 0
    recommendations = [idx_to_hotel_id[i] for i in recommended_idx if not user_rated[i]]

    return recommendations[:top_n]

# --- Cold start recommendation based on user questionnaire ---
def cold_start_recommendation_with_names(location_user, num_cities, num_reviews_profile, num_helpful_votes_user, top_n=10):
    # Step 1: Filter hotels by location (region or locality)
    location_key = location_user.split(",")[0].strip()
    candidate_hotels = df_hotels[
        df_hotels['region'].str.contains(location_key, case=False, na=False) |
        df_hotels['locality'].str.contains(location_key, case=False, na=False)
    ]

    if candidate_hotels.empty:
        candidate_hotels = df_hotels.copy()  # fallback to all hotels

    # Step 2: Compute avg_rating if missing or not present
    if 'avg_rating' not in candidate_hotels.columns or candidate_hotels['avg_rating'].isnull().all():
        avg_ratings = df_reviews.groupby('offering_id')['weighted_score'].mean()
        candidate_hotels = candidate_hotels.merge(avg_ratings.rename('avg_rating'), left_on='offering_id', right_index=True, how='left')
        candidate_hotels['avg_rating'] = candidate_hotels['avg_rating'].fillna(3.5)

    # Debug print unique values
    # print("Unique hotel_class values:", candidate_hotels['hotel_class'].unique())
    # print("Unique avg_rating values:", candidate_hotels['avg_rating'].unique())

    # Step 3: Compute composite score
    user_review_weight = min(num_reviews_profile / 20, 1.0)
    user_helpful_weight = min(num_helpful_votes_user / 20, 1.0)

    # Location relevance: 1 if locality matches exactly (case-insensitive), else 0
    candidate_hotels['location_match'] = candidate_hotels['locality'].str.lower().apply(lambda x: 1.0 if isinstance(x, str) and location_key.lower() in x else 0.0)

    candidate_hotels['score'] = (
        candidate_hotels['hotel_class'].fillna(3.5) * (0.6 + 0.4 * (user_review_weight + user_helpful_weight) / 2) +
        candidate_hotels['avg_rating'] * 0.4 +
        candidate_hotels['location_match'] * 0.5
    )

    if num_cities > 3:
        candidate_hotels.loc[
            (candidate_hotels['hotel_class'] >= 3) & (candidate_hotels['hotel_class'] < 5), 'score'
        ] *= 1.1

    # Step 4: Sort by score
    candidate_hotels = candidate_hotels.sort_values('score', ascending=False).head(top_n)

    # Step 5: Return list of (hotel_name, score) tuples
    recommendations = list(zip(candidate_hotels['name'], candidate_hotels['score']))

    return recommendations


# --- Hybrid recommendation ---
def hybrid_recommend(user_id, top_n=10):
    if user_id in user_id_to_idx:
        # Collaborative recommendations
        collab_recs = recommend_collab(user_id, top_n=top_n*2)

        hotel_indices = [hotel_id_to_idx[h] for h in collab_recs if h in hotel_id_to_idx]

        if len(hotel_indices) == 0:
            # Fallback to collaborative recommendations only if no similar hotels found
            return collab_recs[:top_n]

        # Compute average hotel similarity vector for these recommended hotels
        combined_scores = np.zeros(num_hotels)
        for idx in hotel_indices:
            combined_scores += hotel_similarity[idx].toarray().flatten()
        combined_scores /= len(hotel_indices)

        ranked_hotels = np.argsort(combined_scores)[::-1]

        user_idx = user_id_to_idx[user_id]
        user_rated = user_item_matrix[user_idx].toarray().flatten() > 0

        recommendations = [idx_to_hotel_id[i] for i in ranked_hotels if not user_rated[i]]

        return recommendations[:top_n]

    else:
        # Cold start fallback - prompt user questions
        print("Cold start detected - please answer the following questions:")
        location_user = input("Where do you want to go? ")
        num_cities = int(input("How many different cities do you typically travel to each year? "))
        num_reviews_profile = int(input("Have you reviewed hotels before? (number of reviews) "))
        num_helpful_votes_user = int(input("Do you write detailed helpful reviews? (number of helpful votes) "))

        recs = cold_start_recommendation_with_names(
            location_user,
            num_cities,
            num_reviews_profile,
            num_helpful_votes_user,
            top_n=top_n
        )

        print("\nRecommendations for you:\n")
        for hotel_name, score in recs:
            print(f"{hotel_name}: similarity score {score:.3f}")

        return recs  # returns list of (hotel_name, score)

# --- Example usage ---
user_to_recommend = "4CF3E62850BC1D5449EF4A6D88772EE7"  # known user
print("Hybrid recommendations for existing user:")
existing_recs = hybrid_recommend(user_to_recommend)
print(existing_recs)  # list of hotel IDs

print("\nHybrid recommendations for new user (cold start):")
cold_start_recs = hybrid_recommend("new_user_id")  # triggers cold start, prints names + scores
