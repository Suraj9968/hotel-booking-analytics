import pandas as pd
import sqlite3
import os

# --- 1. Data Collection & Loading ---
# Assuming the hotel_bookings.csv file is in the same directory as your script
df = pd.read_csv('hotel_bookings.csv')

# --- 2. Data Exploration (Optional but Recommended) ---
print("Initial Dataframe Head:")
print(df.head())
print("\nDataframe Info:")
print(df.info())
print("\nMissing Values (before handling):")
print(df.isnull().sum())
print("\nDescriptive Statistics:")
print(df.describe())

# --- 3. Data Cleaning and Preprocessing ---

# a) Handling Missing Values (Modified 'country' imputation)
df['country'].fillna('Unknown', inplace=True)
df['agent'].fillna(0, inplace=True)
df['company'].fillna(0, inplace=True)
df['children'].fillna(0, inplace=True)

# b) Handle Invalid Guest Entries
# Remove bookings with 0 adults, 0 children, and 0 babies
df = df[~( (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0) )]
print(f"\nNumber of rows after removing bookings with 0 total guests: {len(df)}")

# c) Replace "Undefined" with "SC" in 'meal' column
df['meal'].replace('Undefined', 'SC', inplace=True)
print("\nValue counts for 'meal' after replacing 'Undefined' with 'SC':")
print(df['meal'].value_counts())


# d) Data Type Conversion
month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
             'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df['arrival_date_month_num'] = df['arrival_date_month'].map(month_map)

try:
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month_num'].astype(str) + '-' + df['arrival_date_day_of_month'].astype(str))
except ValueError as e:
    print(f"Error creating 'arrival_date': {e}")

df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
df['adr'].fillna(0, inplace=True)

# e) Remove Duplicates (Commented out as per request)
# initial_rows = len(df)
# df.drop_duplicates(inplace=True)
# duplicates_removed = initial_rows - len(df)
# print(f"\nNumber of duplicate rows removed: {duplicates_removed}")
# print(f"Number of rows after removing duplicates: {len(df)}")
print("\nDuplicate removal step skipped as per request.")


# f) Feature Engineering: Calculate 'total_stay_duration' and 'revenue'
df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = df['adr'] * df['total_stay_duration']

# --- 4. Store Processed Data ---

# Create 'data' directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# a) Save Processed DataFrame to CSV in 'data' folder
csv_filepath = os.path.join(data_dir, 'processed_hotel_bookings.csv')
df.to_csv(csv_filepath, index=False)
print(f"\nProcessed data saved to CSV: {csv_filepath}")


# b) Store in SQLite Database in 'data' folder
db_filepath = os.path.join(data_dir, 'hotel_bookings.db')
conn = sqlite3.connect(db_filepath) # Connect to database in 'data' folder

df.to_sql('bookings_table', conn, if_exists='replace', index=False)

# c) Create Indices for Faster Querying (Bonus - Performance)
cursor = conn.cursor()
cursor.execute("CREATE INDEX idx_arrival_date ON bookings_table (arrival_date)")
cursor.execute("CREATE INDEX idx_hotel ON bookings_table (hotel)")
cursor.execute("CREATE INDEX idx_is_canceled ON bookings_table (is_canceled)")
cursor.execute("CREATE INDEX idx_country ON bookings_table (country)")
cursor.execute("CREATE INDEX idx_arrival_year ON bookings_table (arrival_date_year)")
cursor.execute("CREATE INDEX idx_arrival_month_num ON bookings_table (arrival_date_month_num)")

conn.close()

print(f"\nData stored in SQLite database: {db_filepath} in table 'bookings_table' with indices created.")

print("\nMissing Values (after handling):")
print(df.isnull().sum())

print("\nProcessed Dataframe Head (Stored in CSV and DB):")
print(df.head())