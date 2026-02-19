import pickle

def save_model(model, filename="../model/titanic_model.pkl"):

    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")
