# ResumeClassifier12hrs
Resume Ranker
Resume Ranker is a Python-based project designed to streamline the evaluation and ranking of resumes for a Data Scientist position. It combines traditional keyword matching techniques with advanced Natural Language Processing (NLP) by leveraging the Gemini LLM API.

Overview
The project processes resume data from a CSV file, consolidating multiple text fields into a single searchable text block. It then scores each resume using two methods:

Keyword Matching: Counts the frequency of predefined, job-specific keywords (e.g., Python, R, SQL, TensorFlow) in the resume text.
LLM-Based Scoring: Sends the resume and job description to the Gemini LLM API, which acts as an HR expert to provide a match score between 0 and 100.
Key Features
Data Preprocessing: Combines various text segments from resumes for a comprehensive evaluation.
Phone Number Formatting: Corrects phone numbers presented in scientific notation.
Asynchronous Processing: Utilizes asynchronous HTTP requests with aiohttp to efficiently handle multiple API calls concurrently.
Dual Scoring Methods: Offers both a simple keyword-based scoring mechanism and a sophisticated LLM-based evaluation.
CSV Output: Outputs ranked resumes to a CSV file for easy review and further processing.
Usage
Dataset Preparation: Ensure your resume data is in a CSV file, with relevant text segments and phone numbers formatted as strings.
Environment Setup: Set your Gemini LLM API key as an environment variable (GEMINI_API_KEY).
Run the Project: Execute the script to preprocess resumes, calculate scores, and output a ranked CSV file.
This project provides a robust framework for HR professionals and recruiters, offering both quick keyword insights and deeper evaluations through advanced AI, making the candidate selection process more efficient and data-driven.







