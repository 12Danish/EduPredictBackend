from dotenv import load_dotenv
import traceback
import pandas as pd
from google import genai

# Load environment variables from .env file
load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


def generate_suggestions_for_cohort(df):
    """
    Generate suggestions for a cohort of students based on their risk analysis.

    Args:
        df (pandas.DataFrame): DataFrame containing student data with risk analysis

    Returns:
        str: Generated suggestions in markdown format
    """
    print("Generating suggestions for student cohort...")

    # Calculate statistics
    total_students = len(df)
    high_risk_students = df[df['Risk'] == 1]
    low_risk_students = df[df['Risk'] == 0]
    high_risk_count = len(high_risk_students)
    low_risk_count = len(low_risk_students)

    # Calculate averages for high-risk students
    if high_risk_count > 0:
        avg_admission_grade_hr = high_risk_students['Admission grade'].mean()
        avg_1st_sem_grade_hr = high_risk_students['Curricular units 1st sem (grade)'].mean()
        avg_2nd_sem_grade_hr = high_risk_students['Curricular units 2nd sem (grade)'].mean()
        avg_dropout_prob_hr = high_risk_students['Dropout_Probability'].mean()
        unpaid_fees_hr = (high_risk_students['Tuition fees up to date'] == 0).sum()
    else:
        avg_admission_grade_hr = avg_1st_sem_grade_hr = avg_2nd_sem_grade_hr = avg_dropout_prob_hr = unpaid_fees_hr = 0

    # Get sample high-risk student IDs (first 5)
    high_risk_ids = high_risk_students['Student Roll Number'].head(5).tolist() if high_risk_count > 0 else []

    prompt = f"""
    You are an academic advisor analyzing a cohort of {total_students} students for dropout risk.

    ## Cohort Analysis Summary:
    - Total Students: {total_students}
    - High Risk Students: {high_risk_count} ({(high_risk_count / total_students) * 100:.1f}%)
    - Low Risk Students: {low_risk_count} ({(low_risk_count / total_students) * 100:.1f}%)

    ## High-Risk Student Patterns:
    - Average Admission Grade: {avg_admission_grade_hr:.3f}/200
    - Average 1st Semester Grade: {avg_1st_sem_grade_hr:.3f}/20
    - Average 2nd Semester Grade: {avg_2nd_sem_grade_hr:.3f}/20
    - Average Dropout Probability: {avg_dropout_prob_hr:.1%}
    - Students with Unpaid Fees: {unpaid_fees_hr}
    - Sample High-Risk Student IDs: {', '.join(map(str, high_risk_ids))}

    Provide concise, actionable recommendations for instructors to support this cohort:

    1. **Immediate Actions** for high-risk students
    2. **Preventive Measures** for the entire cohort  
    3. **Systemic Improvements** based on the data patterns
    4. **Additional Factors** to investigate (family background, study habits, etc.)

    Format your response in markdown with bullet points. Be practical and specific. 
    Focus on instructor guidance and institutional policies.
    """


    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        response_text = response.text
        print("✅ Successfully generated suggestions")

    except Exception as e:
        print("❌ Failed to generate suggestions")
        print("Exception type:", type(e).__name__)
        print("Exception details:", str(e))
        print("Full traceback:")
        traceback.print_exc()

        # Fallback response
        response_text = "Could not load suggestions"

    return response_text