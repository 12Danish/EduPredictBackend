import traceback

import joblib

def get_dropout_predictions(input_data, model_path="ml_and_llm/student_dropout_model_top8.pkl"):

    try:
        print("model predictions function entered")
        artifacts = joblib.load(model_path)
        model = artifacts['model']
        top_features = artifacts['top_features']
        threshold = artifacts['optimal_threshold']

        print("Before predictions")
        # Selecting the features on which the model was trained
        X = input_data[top_features].copy()



        # Predicting probabilities
        probs = model.predict_proba(X)[:, 1]
        print("Predicted probabilities:", probs)

        # Apply threshold to classify dropout risk
        print("This is the threshold:", threshold)
        preds = probs >= threshold
        print(f"Threshold used: {threshold}")
        print("Predicted dropout risk (True = at risk, False = safe):", preds)

        results = input_data.copy()
        results["Risk"] = preds.astype(int)  # make sure it's 0/1
        results["Dropout_Probability"] = probs

        print("Results DataFrame (sample):")
        print(results.head())

        return results
    except Exception as e:
        print(e)

        traceback.print_exc()


