from data_loader import load_data
from train import train_model
from evaluate import evaluate_model

def main():
    X, y = load_data("data/student_scores.csv")
    model = train_model(X, y)
    mse = evaluate_model(model, X, y)
    
    print("Model trained successfully")
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()