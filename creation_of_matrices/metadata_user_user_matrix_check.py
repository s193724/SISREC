import numpy as np
import scipy.sparse as sp
import json

# === Load similarity matrix ===
similarity_matrix = sp.load_npz("user_similarity_faiss_sparse.npz")
print(f"Loaded similarity matrix with shape: {similarity_matrix.shape}")

# === Load index mappings ===
with open("idx_to_user_id.json", "r") as f:
    idx_to_user_id = json.load(f)

with open("user_id_to_idx.json", "r") as f:
    user_id_to_idx = json.load(f)


# === Function to get top N similar users ===
def get_top_similar_users(user_id, top_n=5):
    if user_id not in user_id_to_idx:
        print(f"User ID {user_id} not found.")
        return []

    user_idx = user_id_to_idx[user_id]
    row = similarity_matrix[user_idx].tocoo()

    # Sort by similarity (descending)
    neighbors = sorted(zip(row.col, row.data), key=lambda x: -x[1])

    results = []
    for idx, sim in neighbors[:top_n]:
        similar_user_id = idx_to_user_id[str(idx)]
        results.append((similar_user_id, sim))

    return results


# === Example: Print top 5 similar users for a given user ID ===
example_user_id = "4CF3E62850BC1D5449EF4A6D88772EE7"  # replace with one from your dataset
similar_users = get_top_similar_users(example_user_id, top_n=5)

print(f"\nTop similar users to {example_user_id}:")
for uid, score in similar_users:
    print(f"  → {uid} (similarity: {score:.4f})")

print(list(user_id_to_idx.keys())[:10])