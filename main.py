import pandas as pd
import re
from decimal import Decimal

# I chose the job profile to be that of a data scientist and assigned keywords
df = pd.read_csv('dataset.csv', dtype={'phone': str})
#fixing some things 
def convert_phone(phone):
    try:
        if isinstance(phone, str) and 'e' in phone.lower():
            return format(Decimal(phone), 'f').rstrip('0').rstrip('.')
        return phone
    except Exception:
        return phone

df['phone'] = df['phone'].apply(convert_phone)

text_columns = [
    'accomplishments_segment', 'degree', 'education_segment', 'job_titles',
    'misc_segment', 'objectives_segment', 'projects_segment', 'skills',
    'skills_segment', 'work_experience', 'work_segment'
]
df['combined_text'] = df[text_columns].fillna('').apply(lambda row: ' '.join([str(item) for item in row]), axis=1)

# Defined keywords for scoring
keywords = [
    "Python", "R", "SQL", "Scala", "Java", "C++",
    "Pandas", "NumPy", "SciPy", "Dask", "Polars",
    "scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "LightGBM", "CatBoost",
    "matplotlib", "seaborn", "ggplot2", "Plotly", "Tableau", "Power BI",
    "PostgreSQL", "MySQL", "MongoDB", "Apache Spark", "Hadoop", "Google BigQuery", "Snowflake",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Flask", "FastAPI", "Streamlit",
    "Git", "GitHub", "GitLab", "Jenkins", "Airflow",
    "Supervised Learning", "Unsupervised Learning", "Feature Engineering", "Hyperparameter Tuning", 
    "Gradient Boosting", "Neural Networks", "Bayesian Statistics", "A/B Testing", "Time Series Analysis", 
    "Clustering", "NLP", "Computer Vision", "Reinforcement Learning", "Deep Learning",
    "Data-driven decision making", "Business intelligence", "Data storytelling", 
    "Experimentation", "hypothesis testing", "ETL", "ML model deployment", "Predictive modeling", 
    "Customer segmentation", "Fraud detection"
]

# Function to score resumes based on keyword frequency
def score_resume(text, keywords):
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        score += len(re.findall(pattern, text_lower))
    return score

df['keyword_score'] = df['combined_text'].apply(lambda text: score_resume(text, keywords))
df_ranked = df.sort_values(by='keyword_score', ascending=False)

print(df_ranked[['name', 'keyword_score', 'phone']].head(10))
df_ranked.to_csv('.csv', index=False)
print("Ranked resumes saved to keyword_resume.csv")
