import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import json


def load_user_features_matrix(path='data_and_RS/user_features_sparse.npz'):
    return sp.load_npz(path)


def save_user_features_matrix(matrix, path='data_and_RS/user_features_sparse.npz'):
    sp.save_npz(path, matrix)


def add_new_user_feature_sparse(existing_matrix, new_user_vector):
    new_user_sparse = sp.csr_matrix(new_user_vector.reshape(1, -1))
    updated_matrix = sp.vstack([existing_matrix, new_user_sparse])
    return updated_matrix


def load_json_mapping(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json_mapping(mapping, path):
    with open(path, 'w') as f:
        json.dump(mapping, f)


def prompt_new_user(total_features):
    print("Please enter your information to personalize recommendations.")

    num_helpful_votes = int(input("Number of helpful votes you have received: "))
    num_cities = int(input("Number of cities you have reviewed: "))
    num_reviews = int(input("Number of reviews you have written: "))

    user_vector = [num_helpful_votes, num_cities, num_reviews]

    padding_length = total_features - len(user_vector)
    if padding_length > 0:
        user_vector.extend([0] * padding_length)
    elif padding_length < 0:
        raise ValueError("New user vector has more features than expected!")

    user_vector = np.array(user_vector, dtype=np.float32)
    user_vector /= (np.linalg.norm(user_vector) + 1e-10)
    return user_vector


def load_user_features_matrix(path='data_and_RS/user_features.npz'):
    sparse = load_npz(path)
    return sparse.toarray()  # convert to dense for easier stacking


def save_user_features_matrix(matrix, path='data_and_RS/user_features.npz'):
    sparse = csr_matrix(matrix)
    save_npz(path, sparse)


def add_new_user_feature(existing_matrix, new_user_vector):
    return np.vstack([existing_matrix, new_user_vector.reshape(1, -1)])


def recompute_metadata_similarity_matrix(user_features_matrix):
    return cosine_similarity(user_features_matrix)


def save_similarity_matrix(matrix, path='data_and_RS/metadata_user_user_similarity.npz'):
    sparse = csr_matrix(matrix)
    save_npz(path, sparse)


def main():
    new_user_vector = prompt_new_user()

    print("Loading existing user features matrix...")
    user_features = load_user_features_matrix()

    print(f"Existing user features shape: {user_features.shape}")
    print("Adding new user feature vector...")
    updated_features = add_new_user_feature(user_features, new_user_vector)
    print(f"Updated features shape: {updated_features.shape}")

    print("Saving updated user features matrix...")
    save_user_features_matrix(updated_features)

    print("Recomputing metadata user-user similarity matrix...")
    sim_matrix = recompute_metadata_similarity_matrix(updated_features)

    print("Saving metadata similarity matrix...")
    save_similarity_matrix(sim_matrix)

    print("Process complete. New user added and similarity matrix updated.")


if __name__ == "__main__":
    main()
