import pandas as pd

df = pd.read_csv('diabetic_data.csv')

# Step 1: Replace '?' with NaN so pandas recognizes them as missing
df.replace('?', pd.NA, inplace=True)

# Step 2: Check actual missing values now
print("Actual Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Step 3: Drop columns with too many missing values (>40% missing)
cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 
                'max_glu_serum', 'A1Cresult']
df.drop(columns=cols_to_drop, inplace=True)
print("\nDropped high-missing columns. New shape:", df.shape)

# Step 4: Drop remaining rows with missing values
df.dropna(inplace=True)
print("Shape after dropping missing rows:", df.shape)

# Step 5: Remove duplicate patients (keep only first visit per patient)
df.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)
print("Shape after removing duplicate patients:", df.shape)

# Step 6: Create binary target variable
# <30 days readmission = 1 (high risk), everything else = 0
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print("\nBinary Target Distribution:\n", df['readmitted_binary'].value_counts())

# Step 7: Save cleaned data
df.to_csv('diabetic_data_cleaned.csv', index=False)
print("\nCleaned data saved successfully!")