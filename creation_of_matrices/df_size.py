import pandas as pd
df = pd.read_csv('USER_DF.csv')
print(df.shape)

df = pd.read_csv('REVIEWS_DF.csv')
print(df.shape)

num_unique_users = df['id_user'].nunique()

print(f"Number of unique users: {num_unique_users}")
# Load the user data_and_RS
df1 = pd.read_csv("USER_DF.csv")

# Count unique users by their ID
num_unique_users1 = df1['id_user'].nunique()

print(f"Number of unique users: {num_unique_users1}")

# Shape of metadata_sim_users: (576689, 576689)
# Shape of colllab_sim_users: (576689, 576689)
# Shape of hotel_user_sparse: (576689, 3945)
# Shape of users features matrix: (576689, 77971)
# Shape of hotel_hotel_sim: (3945, 3945)
# Loaded 576689 users and 3945 hotels from mappings.
# Loaded 576689 users and 4333 hotels.
# New user new_user_123 detected. Onboarding...
# Please enter your information to personalize recommendations.
# Number of helpful votes you have received: 1
# Number of cities you have reviewed: 1
# Number of reviews you have written: 1
# Saving updated data_and_RS to disk...
# Cold start: Using metadata similarity for user new_user_123
# hotel_hotel_sim shape: (3945, 3945)
# user_ratings shape: (1, 3945)
# user_ratings.T shape: (3945, 1)
# Recommendations for new_user_123:
# Hotel ID: hotel_3549, Score: 0.0096
# Hotel ID: hotel_3515, Score: 0.0086
# Hotel ID: hotel_913, Score: 0.0085
# Hotel ID: hotel_3528, Score: 0.0084
# Hotel ID: hotel_617, Score: 0.0080