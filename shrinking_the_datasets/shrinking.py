import pandas as pd

# Load the CSVs
users_df = pd.read_csv("USER_DF.csv")
reviews_df = pd.read_csv("../creation_test_train_and_evaluation/REVIEWS_DF.csv")
hotels_df = pd.read_csv("../creation_test_train_and_evaluation/hotel_df.csv")

reduced_users_df = users_df.sample(frac=0.8, random_state=42)

# 2. Keep only reviews made by those users
filtered_reviews_df = reviews_df[reviews_df["id_user"].isin(reduced_users_df["id_user"])]

# 3. (Optional) Keep only hotels that still have reviews
reviewed_hotels = filtered_reviews_df["offering_id"].unique()
filtered_hotels_df = hotels_df[hotels_df["offering_id"].isin(reviewed_hotels)]

# === Save to new CSVs ===
reduced_users_df.to_csv("USER_DF.csv", index=False)
filtered_reviews_df.to_csv("REVIEWS_DF.csv", index=False)
filtered_hotels_df.to_csv("hotel_df.csv", index=False)

# Logging
print(f"Original users: {len(users_df)}, Reduced to: {len(reduced_users_df)}")
print(f"Original reviews: {len(reviews_df)}, Reduced to: {len(filtered_reviews_df)}")
print(f"Remaining hotels: {len(filtered_hotels_df)}")
