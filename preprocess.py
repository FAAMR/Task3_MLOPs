import pandas as pd
import sys
import os

# URL for the Titanic dataset (raw CSV from GitHub)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

# Load data
try:
    df = pd.read_csv(url)
    print("Titanic dataset loaded successfully!")
except Exception as e:
    print(f"Error loading data from URL: {e}")
    sys.exit(1)

# Preprocessing Steps

# Convert 'Sex' to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode 'Embarked' as dummy variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Drop columns that are not useful for prediction
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Target column is 'Survived' (already 0 or 1)
# Save the preprocessed file
os.makedirs('data', exist_ok=True)
df.to_csv('data/preprocessed_titanic.csv', index=False)
print("Preprocessing complete and file saved.")
