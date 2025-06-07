import random
import pandas as pd

def fabricate_user_mode_users(num_users=100):
    locations = ['New York', 'San Francisco', 'Chicago', 'Los Angeles', 'Miami', 'Seattle', 'Boston', 'Denver']
    users = []

    for i in range(num_users):
        user = {
            'user_id': f'user_{1000 + i}',
            'location': random.choice(locations),
            'cities': round(random.uniform(0.5, 5.0), 2),       # number of cities user is familiar with
            'reviews': random.randint(0, 50),                   # number of reviews user has made
            'helpful': random.randint(0, 20),                   # helpful votes received
            'relevant_hotels': ",".join(random.sample(
                ['121999', '1845691', '781627', '2322597', '93585', '73943', '112052', '87632', '1175274', '1510383'],
                k=random.randint(1, 5)
            ))
        }
        users.append(user)
    return users

def fabricate_hotel_mode_users(num_users=100):
    location_regions = ['NY', 'CA', 'TX', 'IL', 'PA', 'AZ', 'FL', 'IN', 'OH', 'MI', 'NC', 'TN', 'WA', 'MA', 'MD', 'CO', 'DC']
    users = []

    for i in range(num_users):
        user = {
            'user_id': f'user_{2000 + i}',
            'service': round(random.uniform(1.0, 5.0), 2),
            'cleanliness': round(random.uniform(1.0, 5.0), 2),
            'overall': round(random.uniform(1.0, 5.0), 2),
            'value': round(random.uniform(1.0, 5.0), 2),
            'location_pref_score': round(random.uniform(1.0, 5.0), 2),
            'sleep_quality': round(random.uniform(1.0, 5.0), 2),
            'rooms': round(random.uniform(1.0, 5.0), 2),
            'hotel_class': random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
            'location_region': random.choice(location_regions),
            'relevant_hotels': ",".join(random.sample(
                ['121999', '1845691', '781627', '2322597', '93585', '73943', '112052', '87632', '1175274', '1510383'],
                k=random.randint(1, 5)
            ))
        }
        users.append(user)
    return users

def save_users_to_csv(users, filename):
    df = pd.DataFrame(users)
    df.to_csv(filename, index=False)
    print(f"Saved {len(users)} users to {filename}")

# Example usage:
test_users_user_mode = fabricate_user_mode_users(100)
test_users_hotel_mode = fabricate_hotel_mode_users(100)

save_users_to_csv(test_users_user_mode, 'test_users_user_mode.csv')
save_users_to_csv(test_users_hotel_mode, 'test_users_hotel_mode.csv')
