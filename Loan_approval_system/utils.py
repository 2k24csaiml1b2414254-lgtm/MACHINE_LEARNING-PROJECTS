import pickle

# Save trained model to file
def save_model(model, path):
    """
    Saves model as .pkl file
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)


# Load model from file
def load_model(path):
    """
    Loads saved model
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
