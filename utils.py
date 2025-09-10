import pandas as pd

REQUIRED_COLUMNS = [
    "Student Roll Number",
    "Previous qualification (grade)",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Age at enrollment",
    "Tuition fees up to date",
]


def check_and_normalize_values(df):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        raise ValueError(f"Extra columns: {extra_cols}")

    nulls = df.columns[df.isnull().any()].tolist()
    if nulls:
        raise ValueError(f"Columns contain null values: {nulls}")

    if not df['Previous qualification (grade)'].between(0.0, 1.0).all() or not df[
        'Curricular units 1st sem (grade)'].between(0.0, 1.0).all() \
            or not df['Curricular units 2nd sem (grade)'].between(0.0, 1.0).all() or not df['Admission grade'].between(
        0.0, 1.0).all():
        raise ValueError(
            "Previous qualification (grade),Admission grade,Curricular units 1st sem (grade),Curricular units 2nd sem (grade)  must be between 0.0 and 1.0")

    if not df["Tuition fees up to date"].isin([0, 1]).all():
        raise ValueError("Tuition fees up to date must be 0 or 1")

    if not (pd.api.types.is_integer_dtype(df["Age at enrollment"]) and (df["Age at enrollment"] > 0).all()):
        raise ValueError("Age at enrollment must be positive integers")

    if not (
            pd.api.types.is_integer_dtype(df["Curricular units 1st sem (approved)"])
            and (df["Curricular units 1st sem (approved)"] >= 0).all()
    ) or not (
            pd.api.types.is_integer_dtype(df["Curricular units 2nd sem (approved)"])
            and (df["Curricular units 2nd sem (approved)"] >= 0).all()
    ):
        raise ValueError("Curricular units (1st and 2nd sem approved) must be positive integers")

    normalize_values(df)

    return df

def normalize_values(df):
    for col in ["Previous qualification (grade)", "Admission grade"]:
        df[col] = df[col] * 200

    for col in ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
        df[col] = df[col] * 20



