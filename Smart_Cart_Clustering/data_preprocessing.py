import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    return df
