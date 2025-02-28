import pandas as pd
import joblib

def load_model():
    """Load trained model, preprocessor, and label encoders."""
    model = joblib.load("trained_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    selected_features_df = pd.read_csv("selected_features.csv")
    selected_features = selected_features_df["Features"].tolist()
    return model, preprocessor, label_encoders, selected_features

def apply_feature_engineering(input_data):
    """Apply the same feature engineering steps as in training."""
    input_data["EngagementRate"] = input_data["TimeSpentOnCourse"] / (input_data["CompletionRate"] + 1e-6)
    input_data["QuizScorePerAttempt"] = input_data["QuizScores"] / (input_data["NumberOfQuizzesTaken"] + 1e-6)
    input_data["LowEffortUser"] = ((input_data["TimeSpentOnCourse"] < 10) & (input_data["CompletionRate"] < 20)).astype(int)
    return input_data

def make_prediction(input_data, model, preprocessor, label_encoders, selected_features):
    """Preprocess input data and make predictions."""
    input_data = apply_feature_engineering(input_data)
    for col in label_encoders:
        if col in input_data:
            input_data[col] = label_encoders[col].transform(input_data[col])
    input_data = input_data[selected_features]
    processed_data = preprocessor.transform(input_data)

    # Convert processed_data back to a DataFrame
    processed_df = pd.DataFrame(processed_data, columns=selected_features)

    prob = model.predict_proba(processed_df)[:, 1][0]
    return "Completed" if prob >= 0.35 else "Not Completed", prob

if __name__ == "__main__":
    model, preprocessor, label_encoders, selected_features = load_model()
    example_input = pd.DataFrame({
        "CourseCategory": ["Health"],
        "TimeSpentOnCourse": [29.979719],
        "NumberOfVideosWatched": [17],
        "NumberOfQuizzesTaken": [3],
        "QuizScores": [50.365656],
        "CompletionRate": [20.860773]
    })
    prediction, probability = make_prediction(example_input, model, preprocessor, label_encoders, selected_features)
    print(f"Prediction: {prediction} (Probability: {probability:.2f})")