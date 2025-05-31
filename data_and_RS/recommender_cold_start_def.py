import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz, hstack, csr_matrix, vstack
import joblib
import json
import pandas as pd

# --- Load persistent data ---
user_features_sparse = load_npz('user_features_sparse.npz')
user_item_matrix = load_npz('user_hotel_matrix.npz')
user_features_collab = load_npz('user_features_collab.npz')

with open('user_id_to_idx.json') as f:
    user_id_to_idx = json.load(f)
with open('idx_to_user_id.json') as f:
    idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
with open('hotel_idx_to_id.json') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

scaler = joblib.load('scaler.pkl')
vectorizer_location = joblib.load('vectorizer_location.pkl')


def cold_start_recommendation(user_id, top_k=10):
    global user_features_sparse
    global user_item_matrix
    print("Cold start: please answer a few questions.")
    try:
        # Step 1: Input
        location = input("Where do you want to go (location)? ")
        cities = float(input("How many cities do you travel to per year? "))
        reviews = float(input("How many hotel reviews have you written? "))
        helpful = float(input("How many helpful votes have you received? "))

        # Step 2: Build cold user vector
        numeric_vector = scaler.transform([[helpful, cities, reviews]])
        location_vector = vectorizer_location.transform([location])
        cold_user_vector = hstack([csr_matrix(numeric_vector), location_vector])  # shape (1, 503)

        # Step 3: Compute similarity to existing users
        similarities = cosine_similarity(cold_user_vector, user_features_sparse)[0]

        # Step 4: Weighted collaborative recommendation
        scores = similarities @ user_item_matrix
        scores = np.array(scores).flatten()

        # Step 5: Top-K hotels
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_hotels = [(idx_to_hotel_id[i], scores[i]) for i in top_indices]

        # Step 6: Append new user data to persistent structures (optional)
        new_idx = len(user_id_to_idx)
        user_id_to_idx[user_id] = new_idx
        idx_to_user_id[new_idx] = user_id


        user_features_sparse = vstack([user_features_sparse, cold_user_vector])



        user_item_matrix = vstack([user_item_matrix, csr_matrix((1, user_item_matrix.shape[1]))])

        # Save updates (optional - persist later)
        with open('user_id_to_idx.json', 'w') as f:
            json.dump(user_id_to_idx, f)
        with open('idx_to_user_id.json', 'w') as f:
            json.dump({k: v for k, v in idx_to_user_id.items()}, f)

        # Return recommendations
        return top_hotels

    except Exception as e:
        print(f"Error: {e}")
        return []


def cold_start_recommendation_collab(user_id, top_k=10):
    """
    Cold start recommendation using user preferences and hotel features,
    with hard filtering on Number of Rooms and Region.
    """
    # Load hotel metadata matrix and other resources
    hotel_features_sparse = load_npz("hotel_features.npz")

    with open('hotel_idx_to_id.json', 'r') as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

    ohe = joblib.load('hotel_region_ohe.pkl')

    # Constants for feature indices (adjust if your order changes!)
    # [service, cleanliness, overall, value, location, sleep_quality, rooms, weighted_score, region_onehot...]
    hotel_class_idx = 25
    #weighted_score_idx = 7
    region_start_idx = 8

    score_weight_factor = 10

    def safe_float(prompt):
        try:
            val = input(prompt).strip()
            return float(val) if val else 0.0
        except:
            return 0.0

    try:
        # Collect user preferences
        service = safe_float("Service: ")
        cleanliness = safe_float("Cleanliness: ")
        overall = safe_float("Overall: ")
        value = safe_float("Value: ")
        location_pref_score = safe_float("Location quality: ")
        sleep_quality = safe_float("Sleep quality: ")
        rooms = safe_float("Room quality: ")
        hotel_class = safe_float("Hotel class: ")
        #preferred_rooms = int(input("Preferred number of rooms (hard filter): "))

        print("Known regions:", ohe.categories_[0])
        location_region = input("Preferred hotel location (region) (hard filter): ").strip()

        if location_region not in ohe.categories_[0]:
            print(f"Warning: Region '{location_region}' not recognized.")
            if 'Unknown' in ohe.categories_[0]:
                location_region = 'Unknown'
            else:
                location_region = ohe.categories_[0][0]
            print(f"Using region '{location_region}' instead.")

        # Encode region
        location_encoded = ohe.transform([[location_region]]).toarray().flatten()

        # User categories vector
        user_categories = [service, cleanliness, overall, value, location_pref_score, sleep_quality, rooms]

        if any(pd.isna(user_categories)):
            raise ValueError("One or more category preferences are NaN!")

        avg_score = np.mean(user_categories)
        median_hotel_class = 3.0  # or pull from somewhere else if you want
        weighted_score_pref = avg_score * median_hotel_class * score_weight_factor

        # Build full user preference vector with weighted score and region encoding
        user_pref_vector = np.hstack((user_categories, [weighted_score_pref], location_encoded, hotel_class))
        cold_user_vector = csr_matrix(user_pref_vector).reshape(1, -1)

        # Extract rooms and region columns from hotel features
        hotel_class_col = hotel_features_sparse[:, hotel_class_idx].toarray().flatten()
        region_cols = hotel_features_sparse[:, region_start_idx:].toarray()

        # Filter hotels by rooms exact match
        hotel_class_mask = (hotel_class_col == hotel_class)

        # Filter hotels by region exact match
        region_idx_in_ohe = list(ohe.categories_[0]).index(location_region)
        region_mask = (region_cols[:, region_idx_in_ohe] == 1)

        # Combine masks for hard filtering
        combined_mask = region_mask & hotel_class_mask

        # Filter hotel features matrix to filtered hotels only
        hotel_features_filtered = hotel_features_sparse[combined_mask]

        # Compute cosine similarity between user vector and filtered hotel features
        similarities = cosine_similarity(cold_user_vector, hotel_features_filtered)[0]

        # Get top-k indices within filtered hotels
        top_indices_filtered = np.argsort(similarities)[::-1][:top_k]

        # Map filtered indices back to original hotel indices
        original_indices = np.where(combined_mask)[0]
        top_indices_original = original_indices[top_indices_filtered]

        top_hotels = [(idx_to_hotel_id[i], similarities[j]) for j, i in enumerate(top_indices_original)]

        return top_hotels

    except Exception as e:
        print(f"Error: {e}")
        return []
