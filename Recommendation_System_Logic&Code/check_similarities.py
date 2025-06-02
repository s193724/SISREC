import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# --- Define matrix paths ---
paths = {
    "hotel_hotel_sim": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/hotel_similarity_matrix.npz",
    "metadata_user_user": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/metadata_user_user.npz",
    "user_similarity_collab": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/user_similarity_collab.npz",
    "user_features_collab": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/user_features_collab.npz",
    "user_features_sparse": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/user_features_sparse.npz",
    "hotel_user_sparse": "/Users/filiporlikowski/Documents/SISREC_PROJECT/Recommendation_System_Logic&Code/user_hotel_matrix.npz"
}

def describe_sparse_matrix(name, matrix):
    print(f"\n--- {name} ---")
    print(f"Shape: {matrix.shape}")
    print(f"Non-zero elements: {matrix.nnz}")
    print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.2f}%")

    sample_idx = min(5, matrix.shape[0])
    dense_sample = matrix[:sample_idx].toarray()
    print(f"Sample values (first {sample_idx} rows):\n{dense_sample}")

    if matrix.nnz > 0:
        max_val = matrix.max()
        mean_val = matrix.sum() / matrix.nnz
        print(f"Max value: {max_val:.4f}")
        print(f"Mean of non-zero values: {mean_val:.4f}")
    else:
        print("All entries are zero.")

# --- Load and describe all matrices ---
for name, path in paths.items():
    print(f"Loading {name} from: {path}")
    matrix = load_npz(path)
    describe_sparse_matrix(name, matrix)

# --- Optionally, compute similarity manually from user features to check correctness ---
print("\n--- Manual cosine similarity check from user_features_sparse ---")
user_features = load_npz(paths["user_features_sparse"]).toarray()

if user_features.shape[0] >= 2:
    sim = cosine_similarity(user_features[:2])
    print("Cosine similarity between user 0 and 1:", sim[0, 1])
else:
    print("Not enough users for similarity check.")
