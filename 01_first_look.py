import pandas as pd

df = pd.read_csv('diabetic_data.csv')

# Shape of data
print("Shape:", df.shape)

# Column names
print("\nColumns:\n", df.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:\n", df.head())

# Data types
print("\nData Types:\n", df.dtypes)

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Target variable distribution
print("\nReadmission Distribution:\n", df['readmitted'].value_counts())