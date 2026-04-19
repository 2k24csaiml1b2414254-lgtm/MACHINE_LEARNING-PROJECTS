from data_preprocessing import load_data, preprocess_data
from eda import perform_eda
from model import train_model
from utils import save_model

# Load dataset
df = load_data("credit_data.csv")

# Preprocess
df = preprocess_data(df)

# EDA
perform_eda(df)

# Split features and target
X = df.drop("target", axis=1)   # ⚠️ change 'target' to your actual column
y = df["target"]

# Train model
model = train_model(X, y)

# Save model
save_model(model)

print("✅ Pipeline completed successfully!")
