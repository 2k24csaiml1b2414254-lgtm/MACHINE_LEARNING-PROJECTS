import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Function to load dataset
def load_data(path):
    """
    Load CSV data from given path
    """
    return pd.read_csv(path)


# Function to clean and preprocess data
def preprocess_data(df):
    """
    Handles:
    - Missing values
    - Encoding categorical variables
    - Dropping unnecessary columns
    """

    # Drop Loan_ID column (not useful for prediction)
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Separate categorical and numerical columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(exclude='object').columns

    # Fill missing numerical values with mean
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Fill missing categorical values with most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Convert categorical data into numbers using Label Encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df
