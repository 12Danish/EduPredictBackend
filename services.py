import pandas as pd
from utils import check_and_normalize_values

from ml_and_llm.ml_model_predictions import get_dropout_predictions

def student_file_processing_service(file):
    print("Srudent file processing service")
    filename = file.filename.lower()
    print("Filename is: ", filename)

    if filename.endswith('.csv'):
        input_df = pd.read_csv(file)

    elif filename.endswith((".xls", ".xlsx")):
        input_df = pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    # Checking the values of the file uploaded and raising errors if not proper
    check_and_normalize_values(input_df)

    print(input_df.head())

    results_df = get_dropout_predictions(input_df)


    results_df.to_csv("debug_output2.csv", index=False)


