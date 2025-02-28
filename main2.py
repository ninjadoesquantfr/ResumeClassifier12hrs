import os
import pandas as pd
import re
import asyncio
import aiohttp
import json
from decimal import Decimal

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyChE0nBb4ZCxCj7mKzm6rpPxs6kWWcN1W8")

JOB_DESCRIPTION = """Data Scientist

Responsibilities:
- Conduct exploratory data analysis to uncover trends and patterns.
- Build predictive models using statistical and machine learning techniques.
- Visualize and communicate insights to stakeholders.
- Work on end-to-end data projects from data collection to model deployment.

Key Skills:
- Expertise in Python, R, and SQL.
- Experience with data visualization tools (e.g., Tableau, matplotlib).
- Knowledge of statistical analysis and machine learning algorithms.

Technical Skills & Tools:
- Programming Languages: Python, R, SQL, Scala, Java, C++
- Data Analysis & Processing: Pandas, NumPy, SciPy, Dask, Polars
- Machine Learning Libraries: scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, CatBoost
- Data Visualization: matplotlib, seaborn, ggplot2, Plotly, Tableau, Power BI
- Databases & Big Data: PostgreSQL, MySQL, MongoDB, Apache Spark, Hadoop, Google BigQuery, Snowflake
- Cloud & Deployment: AWS, Azure, GCP, Docker, Kubernetes, Flask, FastAPI, Streamlit
- Version Control & CI/CD: Git, GitHub, GitLab, Jenkins, Airflow
- Machine Learning & Statistical Techniques: Supervised Learning, Unsupervised Learning, Feature Engineering, Hyperparameter Tuning, Gradient Boosting, Neural Networks, Bayesian Statistics, A/B Testing, Time Series Analysis, Clustering, NLP, Computer Vision, Reinforcement Learning, Deep Learning
- Soft Skills & Experience: Data-driven decision making, Business intelligence, Data storytelling, Experimentation, hypothesis testing, ETL, ML model deployment, Predictive modeling, Customer segmentation, Fraud detection
"""

df = pd.read_csv('dataset.csv', dtype={'phone': str})

cols = [
    'accomplishments_segment', 'degree', 'education_segment', 'job_titles',
    'misc_segment', 'objectives_segment', 'projects_segment', 'skills',
    'skills_segment', 'work_experience', 'work_segment'
]
df['combined_text'] = df[cols].fillna('').apply(lambda row: ' '.join(str(x) for x in row), axis=1)

def format_phone(phone):
    try:
        if isinstance(phone, str) and 'e' in phone.lower():
            return format(Decimal(phone), 'f').rstrip('0').rstrip('.')
        return phone
    except Exception:
        return phone

df['phone'] = df['phone'].apply(format_phone)

async def get_similarity_score(session, resume_text, job_desc, api_key):
    prompt = (
        "You are an HR expert. Evaluate the following resume for the given job description. "
        "Provide a score between 0 and 100, where 100 is a perfect match. "
        "Output your answer strictly as a JSON object with the format: {\"score\": NUMBER}.\n\n"
        f"Job Description: {job_desc}\n\nResume: {resume_text}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 20, "temperature": 0.0}
    url = "https://api.gemini.com/v1/llm"
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                text = data.get("text", "")
                try:
                    parsed = json.loads(text)
                    if "score" in parsed:
                        return float(parsed["score"])
                except Exception:
                    m = re.search(r'(\d+(\.\d+)?)', text)
                    if m:
                        return float(m.group(1))
            return 0.0
    except Exception:
        return 0.0

async def process_resumes(df, api_key, job_desc):
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(5)
        async def worker(text):
            async with sem:
                return await get_similarity_score(session, text, job_desc, api_key)
        tasks = [asyncio.create_task(worker(text)) for text in df['combined_text']]
        return await asyncio.gather(*tasks)

async def main():
    scores = await process_resumes(df, GEMINI_API_KEY, JOB_DESCRIPTION)
    df['gemini_score'] = scores
    df_ranked = df.sort_values(by='gemini_score', ascending=False)
    df_ranked.to_csv('llm_resume.csv', index=False)
    print("Ranked resumes saved to llm_resume.csv")
    print(df_ranked[['name', 'gemini_score', 'phone']].head(10))

if __name__ == "__main__":
    asyncio.run(main())
