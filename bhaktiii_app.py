import os
import pickle

def load_model():
    """Loads the pickled model from the specified file."""
    model_path = os.path.abspath("mymodel.pkl")  # Get absolute path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    try:
        # Explicitly open in binary mode
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return clf
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"Error loading model: {e}") from e

# Example usage in your Streamlit app
try:
    clf = load_model()
except RuntimeError as e:
    print(f"Error loading model: {e}")
else:
    # Use the loaded model for predictions
    # ...
