from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model
from src.utils import save_model

def main():
    """
    Main pipeline:
    1. Load data
    2. Preprocess
    3. Train model
    4. Save model
    """

    # Step 1: Load dataset
    df = load_data("data/loan_approval_data.csv")

    # Step 2: Preprocess data
    df = preprocess_data(df)

    # Step 3: Train model
    model = train_model(df, target_column='Loan_Status')

    # Step 4: Save trained model
    save_model(model, "models/loan_model.pkl")

    print("✅ Model training completed and saved!")

# Run project
if __name__ == "__main__":
    main()
