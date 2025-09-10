import pandas as pd
from utils import check_and_normalize_values

from ml_and_llm.ml_model_predictions import get_dropout_predictions
from ml_and_llm.llm_suggestions import generate_suggestions_for_cohort


def student_file_processing_service(file):
    print("Student file processing service")
    filename = file.filename.lower()
    print("Filename is: ", filename)

    if filename.endswith('.csv'):
        input_df = pd.read_csv(file)

    elif filename.endswith((".xls", ".xlsx")):
        input_df = pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    # Checking the values of the file uploaded and raising errors if not proper
    normalised_df = check_and_normalize_values(input_df.copy())

    print(input_df.head())

    results_df = get_dropout_predictions(normalised_df)

    suggestions = generate_suggestions_for_cohort(results_df)

    results_subset = results_df[["Student Roll Number", "Risk", "Dropout_Probability"]]

    # merge with input_df on Student Roll Number
    final_df = pd.merge(
        input_df,
        results_subset,
        on="Student Roll Number",
        how="left"  # keeps all rows from input_df
    )

    # optional: save for debugging
    final_df.to_csv("debug_output2.csv", index=False)

    return {
        'final_df': final_df.to_dict(orient="records"),
        'suggestions': suggestions
    }

