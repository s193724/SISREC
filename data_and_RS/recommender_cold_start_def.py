import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import load_npz, hstack, csr_matrix, vstack
import joblib
import json

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
    global user_features_collab
    global user_item_matrix

    with open('user_id_to_idx.json', 'r') as f:
        user_id_to_idx = json.load(f)
    with open('idx_to_user_id.json', 'r') as f:
        idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
    with open('hotel_idx_to_id.json', 'r') as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

    ohe = joblib.load('hotel_region_ohe.pkl')

    # Assume median hotel class from your dataset or domain knowledge
    median_hotel_class = 3.0

    print("Please rate your preferences for the following hotel features (1 = low, 5 = high):")
    try:
        service = float(input("Service: "))
        cleanliness = float(input("Cleanliness: "))
        overall = float(input("Overall: "))
        value = float(input("Value: "))
        location_pref_score = float(input("Location quality: "))
        sleep_quality = float(input("Sleep quality: "))
        rooms = float(input("Room quality: "))

        # Average user preference score
        avg_score = np.mean([service, cleanliness, overall, value, location_pref_score, sleep_quality, rooms])
        # Weighted score by median hotel class
        weighted_score_pref = avg_score * median_hotel_class

        # Get user location region
        location_region = input("Preferred hotel location (region): ")
        location_encoded = ohe.transform([[location_region]]).toarray().flatten()

        # Build user feature vector: weighted score + region one-hot vector
        user_pref_vector = np.hstack(([weighted_score_pref], location_encoded))

        cold_user_vector = csr_matrix(user_pref_vector).reshape(1, -1)

        similarities = cosine_similarity(cold_user_vector, user_features_collab)[0]

        scores = similarities @ user_item_matrix
        scores = np.array(scores).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_hotels = [(idx_to_hotel_id[i], scores[i]) for i in top_indices]

        # Update mappings and matrices
        new_idx = len(user_id_to_idx)
        user_id_to_idx[user_id] = new_idx
        idx_to_user_id[new_idx] = user_id

        user_features_collab = vstack([user_features_collab, cold_user_vector])
        user_item_matrix = vstack([user_item_matrix, csr_matrix((1, user_item_matrix.shape[1]))])

        with open('user_id_to_idx.json', 'w') as f:
            json.dump(user_id_to_idx, f)
        with open('idx_to_user_id.json', 'w') as f:
            json.dump({k: v for k, v in idx_to_user_id.items()}, f)

        return top_hotels

    except Exception as e:
        print(f"Error: {e}")
        return []

