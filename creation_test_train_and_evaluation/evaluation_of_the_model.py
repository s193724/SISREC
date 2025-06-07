import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import json
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz, hstack, csr_matrix, vstack

from Recommendation_System_Logic_Code.recommender_cold_start_def import cold_start_recommendation_combined

base_dir = '/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic_Code/'

with open(f'{base_dir}user_id_to_idx.json') as f:
    user_id_to_idx = json.load(f)
with open(f'{base_dir}idx_to_user_id.json') as f:
    idx_to_user_id = {int(k): v for k, v in json.load(f).items()}
with open(f'{base_dir}hotel_id_to_idx.json') as f:
    hotel_id_to_idx = json.load(f)
with open(f'{base_dir}hotel_idx_to_id.json') as f:
    idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

test_users_user_mode = pd.read_csv('test_users_user_mode.csv')
test_users_hotel_mode = pd.read_csv('test_users_hotel_mode.csv')
user_similarity = load_npz(f'{base_dir}user_similarity_collab.npz')
hotel_similarity = load_npz(f'{base_dir}hotel_similarity_matrix.npz')


def get_non_personalized_recommendations(top_k: int = 10, diversify: bool = False):
    hotel_meta_df = pd.read_csv(f'{base_dir}hotel_df.csv')

    hotel_features_sparse = load_npz(f'{base_dir}/hotel_features.npz')
    hotel_similarity = load_npz(f'{base_dir}/hotel_similarity_matrix.npz')

    df = hotel_meta_df.copy()

    with open(f"{base_dir}/hotel_idx_to_id.json") as f:
        idx_to_hotel_id = {int(k): v for k, v in json.load(f).items()}

    with open(f"{base_dir}/hotel_id_to_idx.json") as f:
        hotel_id_to_idx = {k: int(v) for k, v in json.load(f).items()}

    idx_to_hotel_meta = {}
    for idx, row in df.iterrows():
        hotel_id = str(row['offering_id'])  # replace 'hotel_id' with your actual column name
        if hotel_id in hotel_id_to_idx:
            hotel_idx = hotel_id_to_idx[hotel_id]
            idx_to_hotel_meta[hotel_idx] = (hotel_id, row)
        else:
            print(f"Hotel ID {hotel_id} not found in hotel_id_to_idx")

    weighted_score_col = 7
    hotel_class_col = -1

    weighted_scores = hotel_features_sparse[:, weighted_score_col].toarray().flatten()
    hotel_classes = hotel_features_sparse[:, hotel_class_col].toarray().flatten()

    print(f"Loaded weighted_scores shape: {weighted_scores.shape}")
    print(f"Loaded hotel_classes shape: {hotel_classes.shape}")

    norm_scores = (weighted_scores - weighted_scores.min()) / (weighted_scores.max() - weighted_scores.min())
    norm_class = (hotel_classes - hotel_classes.min()) / (hotel_classes.max() - hotel_classes.min())

    combined_scores = 0.6 * norm_scores + 0.4 * norm_class

    if diversify:
        top_indices = np.argsort(combined_scores)[::-1][:20]
        sim_scores = hotel_similarity[top_indices].mean(axis=0).A1
        combined_scores += 0.1 * sim_scores

    final_top_indices = np.argsort(combined_scores)[::-1][:top_k]

    print(f"Top indices: {final_top_indices}")
    print(f"Top combined scores: {combined_scores[final_top_indices]}")

    recommendations = []
    for idx in final_top_indices:
        if idx in idx_to_hotel_meta:
            hotel_id, meta_row = idx_to_hotel_meta[idx]
            recommendations.append((hotel_id, combined_scores[idx]))
        else:
            print(f"Index {idx} not found in idx_to_hotel_meta")
    print(f"hotel_features_sparse shape: {hotel_features_sparse.shape}")
    print(f"hotel_similarity shape: {hotel_similarity.shape}")
    print(f"Number of hotels in idx_to_hotel_meta: {len(idx_to_hotel_meta)}")
    print(f"Top combined scores (first 10): {combined_scores[:10]}")
    print(f"final_top_indices: {final_top_indices}")
    print(f"Recommendations collected: {len(recommendations)}")
    print(f"Number of recommendations: {len(recommendations)}")
    print("Sample hotel_meta_df index (hotel ids):", list(df.index[:10]))
    print("Sample hotel_id_to_idx keys:", list(hotel_id_to_idx.keys())[:10])
    return recommendations

scaler = joblib.load(f'{base_dir}scaler.pkl')
vectorizer_location = joblib.load(f'{base_dir}vectorizer_location.pkl')

def cold_start_recommendation_combined(
    user_id,
    mode="user",
    top_k=10,
    # User mode params
    location=None,
    cities=None,
    reviews=None,
    helpful=None,
    # Hotel mode params
    service=None,
    cleanliness=None,
    overall=None,
    value=None,
    location_pref_score=None,
    sleep_quality=None,
    rooms=None,
    hotel_class=None,
    location_region=None
):
    user_features_sparse = load_npz(f'{base_dir}user_features_sparse.npz')
    user_item_matrix = load_npz(f'{base_dir}user_hotel_matrix.npz')

    try:
        if mode == "user":
            # Validate required params
            if None in [location, cities, reviews, helpful]:
                raise ValueError("Missing parameters for user mode")

            numeric_vector = scaler.transform([[helpful, cities, reviews]])
            location_vector = vectorizer_location.transform([location])
            cold_user_vector = hstack([csr_matrix(numeric_vector), location_vector])

            similarities = cosine_similarity(cold_user_vector, user_features_sparse)[0]
            scores = similarities @ user_item_matrix
            scores = np.array(scores).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_hotels = [(idx_to_hotel_id[i], scores[i]) for i in top_indices]

            # Update persistent data
            new_idx = len(user_id_to_idx)
            user_id_to_idx[user_id] = new_idx
            idx_to_user_id[new_idx] = user_id

            user_features_sparse = vstack([user_features_sparse, cold_user_vector])
            user_item_matrix = vstack([user_item_matrix, csr_matrix((1, user_item_matrix.shape[1]))])

            with open('user_id_to_idx.json', 'w') as f:
                json.dump(user_id_to_idx, f)
            with open('idx_to_user_id.json', 'w') as f:
                json.dump({k: v for k, v in idx_to_user_id.items()}, f)

            return top_hotels

        elif mode == "hotel":
            # Validate required params
            hotel_mode_params = [service, cleanliness, overall, value, location_pref_score,
                                sleep_quality, rooms, hotel_class, location_region]
            if any(p is None for p in hotel_mode_params):
                raise ValueError("Missing parameters for hotel mode")

            hotel_features_sparse = load_npz(f'{base_dir}hotel_features.npz')
            ohe = joblib.load(f'{base_dir}hotel_region_ohe.pkl')

            if location_region not in ohe.categories_[0]:
                print(f"Region '{location_region}' not recognized.")
                location_region = 'Unknown' if 'Unknown' in ohe.categories_[0] else ohe.categories_[0][0]
                print(f"Using region '{location_region}' instead.")

            location_encoded = ohe.transform([[location_region]]).toarray().flatten()

            user_categories = [service, cleanliness, overall, value, location_pref_score, sleep_quality, rooms]
            avg_score = np.mean(user_categories)
            median_hotel_class = 3.0
            weighted_score_pref = avg_score * median_hotel_class * 10  # weight factor

            user_pref_vector = np.hstack((user_categories, [weighted_score_pref], location_encoded, hotel_class))
            cold_user_vector = csr_matrix(user_pref_vector).reshape(1, -1)

            hotel_class_idx = 25
            region_start_idx = 8

            hotel_class_col = hotel_features_sparse[:, hotel_class_idx].toarray().flatten()
            region_cols = hotel_features_sparse[:, region_start_idx:].toarray()

            hotel_class_mask = (hotel_class_col == hotel_class)
            region_idx = list(ohe.categories_[0]).index(location_region)
            region_mask = (region_cols[:, region_idx] == 1)

            combined_mask = region_mask & hotel_class_mask
            hotel_features_filtered = hotel_features_sparse[combined_mask]

            if hotel_features_filtered.shape[0] == 0:
                print("No matching hotels found for selected filters.")
                return []

            similarities = cosine_similarity(cold_user_vector, hotel_features_filtered)[0]
            top_indices_filtered = np.argsort(similarities)[::-1][:top_k]
            original_indices = np.where(combined_mask)[0]
            top_indices_original = original_indices[top_indices_filtered]

            top_hotels = [(idx_to_hotel_id[i], similarities[j]) for j, i in enumerate(top_indices_original)]
            return top_hotels

        else:
            print("Invalid mode. Use 'user' or 'hotel'.")
            return []

    except Exception as e:
        print(f"Error: {e}")
        return []
import csv

def load_test_users_user_mode(csv_path):
    test_users = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_users.append({
                'user_id': row['user_id'],
                'location': row['location'],
                'cities': float(row['cities']),
                'reviews': float(row['reviews']),
                'helpful': float(row['helpful']),
                'relevant_hotels': row['relevant_hotels'].split('|') if row['relevant_hotels'] else []
            })
    return test_users

def load_test_users_hotel_mode(csv_path):
    test_users = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_users.append({
                'user_id': row['user_id'],
                'service': float(row['service']),
                'cleanliness': float(row['cleanliness']),
                'overall': float(row['overall']),
                'value': float(row['value']),
                'location_pref_score': float(row['location_pref_score']),
                'sleep_quality': float(row['sleep_quality']),
                'rooms': float(row['rooms']),
                'hotel_class': float(row['hotel_class']),
                'location_region': row['location_region'],
                'relevant_hotels': row['relevant_hotels'].split('|') if row['relevant_hotels'] else []
            })
    return test_users


city_penalty = {
    "New York City": 0.6,
    "Houston": 0.85,
    "San Antonio": 0.9,
    # Add others or default = 1.0
}

def apply_city_penalty(recommendations):
    global hotel_meta_df
    adjusted = []
    for hotel_id, score in recommendations:
        if hotel_id in hotel_meta_df.index:
            city = hotel_meta_df.loc[hotel_id]['locality']
            penalty = city_penalty.get(city, 1.0)
            adjusted.append((hotel_id, score * penalty))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted[:10]

def evaluate_cold_start_hotel_mode(test_users, top_k=10):
    precisions = []
    recalls = []

    for user in test_users:
        user_id = user['user_id']
        relevant_hotels = set(user.get('relevant_hotels', []))

        recommendations = cold_start_recommendation_combined(
            user_id=user_id,
            mode='hotel',
            top_k=top_k,
            service=user.get('service'),
            cleanliness=user.get('cleanliness'),
            overall=user.get('overall'),
            value=user.get('value'),
            location_pref_score=user.get('location_pref_score'),
            sleep_quality=user.get('sleep_quality'),
            rooms=user.get('rooms'),
            hotel_class=user.get('hotel_class'),
            location_region=user.get('location_region')
        )

        recommended_hotels = [hotel_id for hotel_id, score in recommendations]
        hits = len(set(recommended_hotels) & relevant_hotels)

        precision = hits / top_k if top_k > 0 else 0
        recall = hits / len(relevant_hotels) if relevant_hotels else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    print(f"Avg Precision@{top_k} for hotel mode cold start: {avg_precision:.4f}")
    print(f"Avg Recall@{top_k} for hotel mode cold start: {avg_recall:.4f}")

    return avg_precision, avg_recall


def evaluate_cold_start_user_mode(test_users, top_k=10):
    precisions = []
    recalls = []

    for user in test_users:
        user_id = user['user_id']
        location = user['location']
        cities = user['cities']
        reviews = user['reviews']
        helpful = user['helpful']

        relevant_hotels = user['relevant_hotels']  # list of hotel ids the user interacted with

        recommendations = cold_start_recommendation_combined(
            user_id=user_id,
            mode='user',
            top_k=top_k,
            location=location,
            cities=cities,
            reviews=reviews,
            helpful=helpful
        )
        recommended_ids = [hid for hid, score in recommendations]

        hits = len(set(recommended_ids) & set(relevant_hotels))
        precision = hits / top_k if top_k > 0 else 0
        recall = hits / len(relevant_hotels) if relevant_hotels else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    print(f"Avg Precision@{top_k}: {avg_precision:.4f}")
    print(f"Avg Recall@{top_k}: {avg_recall:.4f}")



def hybrid_recommend(user_id, user_item_matrix, user_similarity, hotel_similarity,
                     user_id_to_idx, idx_to_hotel_id,
                     alpha=0.7, top_k=10):

    if user_id not in user_id_to_idx:
        # Default cold start mode or skip
        mode = 'user'
        return cold_start_recommendation_combined(user_id, mode, top_k=top_k)

    user_idx = user_id_to_idx[user_id]

    user_sim_scores = user_similarity[user_idx].toarray().flatten()
    if np.sum(user_sim_scores) > 0:
        collab_scores = (user_sim_scores @ user_item_matrix) / np.sum(user_sim_scores)
    else:
        collab_scores = np.zeros(user_item_matrix.shape[1])

    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    item_sim_scores = user_ratings @ hotel_similarity

    collab_scores = normalize(collab_scores.reshape(1, -1))[0]
    item_sim_scores = normalize(item_sim_scores.reshape(1, -1))[0]

    hybrid_scores = alpha * collab_scores + (1 - alpha) * item_sim_scores

    rated_indices = np.where(user_ratings > 0)[0]
    hybrid_scores[rated_indices] = 0

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    recommended_hotel_ids = [idx_to_hotel_id[i] for i in top_indices]
    recommended_scores = [hybrid_scores[i] for i in top_indices]

    return list(zip(recommended_hotel_ids, recommended_scores))


def precision_recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = len(set(recommended_k) & relevant_set)
    precision = hits / k if k > 0 else 0
    recall = hits / len(relevant) if len(relevant) > 0 else 0
    return precision, recall


def evaluate_model(user_item_train_path, user_item_test_path, user_id_to_idx, idx_to_user_id,
                   hotel_idx_to_id, user_similarity, hotel_similarity,
                   top_k=10, alpha=0.7):

    user_item_train = load_npz(user_item_train_path)
    user_item_test = load_npz(user_item_test_path)

    num_users = user_item_test.shape[0]

    precisions, recalls = [], []
    users_evaluated = 0

    for user_idx in range(num_users):
        if user_idx not in idx_to_user_id:
            continue
        user_id = idx_to_user_id[user_idx]

        test_item_indices = user_item_test[user_idx].indices
        if len(test_item_indices) == 0:
            continue

        test_item_ids = [hotel_idx_to_id[i] for i in test_item_indices]

        recommendations = hybrid_recommend(
            user_id,
            user_item_train,
            user_similarity,
            hotel_similarity,
            user_id_to_idx,
            idx_to_hotel_id,
            alpha=alpha,
            top_k=top_k
        )
        recommended_items = [hotel_id for hotel_id, _ in recommendations]

        prec, rec = precision_recall_at_k(recommended_items, test_item_ids, top_k)
        precisions.append(prec)
        recalls.append(rec)
        users_evaluated += 1

    avg_precision = np.mean(precisions) if users_evaluated > 0 else 0
    avg_recall = np.mean(recalls) if users_evaluated > 0 else 0

    print(f"Evaluated {users_evaluated} users")
    print(f"Average Precision@{top_k}: {avg_precision:.4f}")
    print(f"Average Recall@{top_k}: {avg_recall:.4f}")

    return avg_precision, avg_recall

def evaluate_non_personalized(
    user_item_test_path,
    hotel_idx_to_id,
    top_k=10,
    diversify=False
):
    user_item_test = load_npz(user_item_test_path)
    num_users = user_item_test.shape[0]

    # Generate global non-personalized recommendations once
    recommendations = get_non_personalized_recommendations(top_k=top_k, diversify=diversify)
    recommended_hotel_ids = [hotel_id for hotel_id, score in recommendations]

    precisions = []
    recalls = []
    users_evaluated = 0

    for user_idx in range(num_users):
        test_items_indices = user_item_test[user_idx].indices
        if len(test_items_indices) == 0:
            continue

        relevant_hotels = [hotel_idx_to_id[i] for i in test_items_indices]

        prec, rec = precision_recall_at_k(recommended_hotel_ids, relevant_hotels, top_k)
        precisions.append(prec)
        recalls.append(rec)
        users_evaluated += 1

        print(f"User index: {user_idx}")
        print(f"Recommended IDs: {recommended_hotel_ids[:top_k]}")
        print(f"Relevant IDs: {relevant_hotels}")
        print(f"Precision@{top_k}, Recall@{top_k}: {prec:.4f}, {rec:.4f}\n")

    avg_precision = np.mean(precisions) if users_evaluated > 0 else 0
    avg_recall = np.mean(recalls) if users_evaluated > 0 else 0

    print(f"Evaluated {users_evaluated} users")
    print(f"Average Precision@{top_k}: {avg_precision:.4f}")
    print(f"Average Recall@{top_k}: {avg_recall:.4f}")

    return avg_precision, avg_recall

def hybrid_recommend_with_city_penalty(user_id, user_item_matrix, user_similarity, hotel_similarity,
                                      user_id_to_idx, idx_to_hotel_id, hotel_meta_df,
                                      alpha=0.7, top_k=10, city_penalty=None):
    # Call original hybrid recommend
    recs = hybrid_recommend(user_id, user_item_matrix, user_similarity, hotel_similarity,
                           user_id_to_idx, idx_to_hotel_id, alpha=alpha, top_k=top_k*3)  # get more to allow filtering

    if city_penalty is None:
        city_penalty = {"New York City": 0.6, "Houston": 0.85, "San Antonio": 0.9}

    adjusted_recs = []
    for hotel_id, score in recs:
        if hotel_id in hotel_meta_df.index:
            city = hotel_meta_df.loc[hotel_id]['locality']
            penalty = city_penalty.get(city, 1.0)
            adjusted_recs.append((hotel_id, score * penalty))
        else:
            # If no city info, no penalty
            adjusted_recs.append((hotel_id, score))
    # Sort by adjusted score
    adjusted_recs.sort(key=lambda x: x[1], reverse=True)

    return adjusted_recs[:top_k]

def evaluate_hybrid_with_city_penalty(user_item_train_path, user_item_test_path, user_id_to_idx, idx_to_user_id,
                                     hotel_idx_to_id, user_similarity, hotel_similarity, hotel_meta_df,
                                     top_k=10, alpha=0.7, city_penalty=None):

    user_item_train = load_npz(user_item_train_path)
    user_item_test = load_npz(user_item_test_path)

    num_users = user_item_test.shape[0]

    precisions, recalls = [], []
    users_evaluated = 0

    for user_idx in range(num_users):
        if user_idx not in idx_to_user_id:
            continue
        user_id = idx_to_user_id[user_idx]

        test_item_indices = user_item_test[user_idx].indices
        if len(test_item_indices) == 0:
            continue

        test_item_ids = [hotel_idx_to_id[i] for i in test_item_indices]

        recommendations = hybrid_recommend_with_city_penalty(
            user_id,
            user_item_train,
            user_similarity,
            hotel_similarity,
            user_id_to_idx,
            hotel_idx_to_id,
            hotel_meta_df,
            alpha=alpha,
            top_k=top_k,
            city_penalty=city_penalty
        )
        recommended_items = [hotel_id for hotel_id, _ in recommendations]

        prec, rec = precision_recall_at_k(recommended_items, test_item_ids, top_k)
        precisions.append(prec)
        recalls.append(rec)
        users_evaluated += 1

    avg_precision = np.mean(precisions) if users_evaluated > 0 else 0
    avg_recall = np.mean(recalls) if users_evaluated > 0 else 0

    print(f"Evaluated {users_evaluated} users")
    print(f"Average Precision@{top_k} with city penalty: {avg_precision:.4f}")
    print(f"Average Recall@{top_k} with city penalty: {avg_recall:.4f}")

    return avg_precision, avg_recall


if __name__ == "__main__":
    user_item_train_path = 'user_hotel_matrix_train.npz'
    user_item_test_path = 'user_hotel_matrix_test.npz'

#increasing alpha means more collaborative filtering
#decreasing alpha means more content-based
    # avg_prec, avg_rec = evaluate_model(       #no city penalty
    #     user_item_train_path,
    #     user_item_test_path,
    #     user_id_to_idx,
    #     idx_to_user_id,
    #     idx_to_hotel_id,
    #     user_similarity,
    #     hotel_similarity,
    #     top_k=10,
    #     alpha=0.7
    # )
    #Evaluated 15495 users
# Average Precision@10: 0.1077
# Average Recall@10: 0.9726

# avg_prec, avg_rec = evaluate_model(
#         user_item_train_path,
#         user_item_test_path,
#         user_id_to_idx,
#         idx_to_user_id,
#         idx_to_hotel_id,
#         user_similarity,
#         hotel_similarity,
#         top_k=10,
#         alpha=0.5
#     )
#
# Average Precision@10: 0.1071
# Average Recall@10: 0.9676

# avg_prec, avg_rec = evaluate_model(
#         user_item_train_path,
#         user_item_test_path,
#         user_id_to_idx,
#         idx_to_user_id,
#         idx_to_hotel_id,
#         user_similarity,
#         hotel_similarity,
#         top_k=10,
#         alpha=0.1
#     )
# Evaluated 15495 users
# Average Precision@10: 0.1011
# Average Recall@10: 0.9234

# avg_prec, avg_rec = evaluate_model(
#         user_item_train_path,
#         user_item_test_path,
#         user_id_to_idx,
#         idx_to_user_id,
#         idx_to_hotel_id,
#         user_similarity,
#         hotel_similarity,
#         top_k=10,
#         alpha=5
#     )
# Evaluated 15495 users
# Average Precision@10: 0.1074
# Average Recall@10: 0.9708

# avg_prec, avg_rec = evaluate_non_personalized(
#         user_item_test_path,
#         idx_to_hotel_id,
#         top_k=10,
#         diversify=False
#     )
# Evaluated 15495 users
# Average Precision@10: 0.0000
# Average Recall@10: 0.0000



# test_users_user_mode = load_test_users_user_mode('test_users_user_mode.csv')
# test_users_hotel_mode = load_test_users_hotel_mode('test_users_hotel_mode.csv')
#
# # Call evaluation functions
# evaluate_cold_start_user_mode(test_users_user_mode, top_k=10)
# evaluate_cold_start_hotel_mode(test_users_hotel_mode, top_k=10)


city_penalty = {
        "New York City": 0.6,
        "Houston": 0.85,
        "San Antonio": 0.9,
    }

# evaluate_hybrid_with_city_penalty(
#         user_item_train_path,
#         user_item_test_path,
#         user_id_to_idx,
#         idx_to_user_id,
#         idx_to_hotel_id,
#         user_similarity,
#         hotel_similarity,
#         hotel_meta_df = pd.read_csv(f'{base_dir}hotel_df.csv'),
#         top_k=10,
#         alpha=0.7,
#         city_penalty=city_penalty
#     )
# Average Precision@10 with city penalty: 0.1077
# Average Recall@10 with city penalty: 0.9726

# evaluate_hybrid_with_city_penalty(
#         user_item_train_path,
#         user_item_test_path,
#         user_id_to_idx,
#         idx_to_user_id,
#         idx_to_hotel_id,
#         user_similarity,
#         hotel_similarity,
#         hotel_meta_df = pd.read_csv(f'{base_dir}hotel_df.csv'),
#         top_k=10,
#         alpha=0.2,
#         city_penalty=city_penalty
#     )
# Average Precision@10 with city penalty: 0.1043
# Average Recall@10 with city penalty: 0.9472