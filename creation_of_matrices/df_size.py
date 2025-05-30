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