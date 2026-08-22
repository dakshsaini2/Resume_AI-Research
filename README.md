# Resume_AI-Research
# 🧠 Resume AI — Intelligent ATS Resume Screening System. 

An AI-powered **Applicant Tracking System (ATS)** that analyzes resumes against job descriptions and generates an intelligent compatibility score using **NLP, semantic similarity, TF-IDF, skill matching, and a Random Forest machine learning model**.

The system helps recruiters and candidates understand how well a resume matches a particular job and identifies missing skills, experience gaps, education compatibility, and other important factors.

---

## 🚀 Features 

### 📄 Resume Analysis

* Supports **PDF** and **DOCX** resumes
* Extracts resume text automatically
* Detects:

  * Skills
  * Education
  * Experience
  * Certifications
  * Relevant candidate information

### 💼 Job Description Analysis

The system analyzes the provided job description and extracts:

* Required skills
* Educational requirements
* Experience requirements
* Certification requirements
* Job-related information

### 🎯 ATS Match Score

Generates an overall resume-job compatibility score based on multiple features.

Example:

```text
ATS Match Score
      78.8%
   Strong Match
```

### 🛠 Skill Matching

The system identifies:

* ✅ Matched skills
* ❌ Missing skills
* 🎯 Overall skill coverage

Example:

```text
Matched Skills:
✓ Python
✓ SQL
✓ Machine Learning

Missing Skills:
+ Docker
+ AWS
+ Kubernetes
```

### 🎓 Education Matching

Compares the candidate's educational qualification with the requirements of the job.

### 💼 Experience Matching

Extracts professional experience from resume text and compares it with the required experience.

The system supports formats such as:

```text
3 years experience
2+ years
6 months
8 weeks
May 2022 to June 2022
Sep 2024 - Dec 2024
```

### 📜 Certification Analysis

Detects certifications from the resume and compares them against job requirements.

### 🧠 Semantic Similarity

Uses transformer-based sentence embeddings to measure the semantic relationship between the resume and job description.

### 📊 TF-IDF Similarity

Uses TF-IDF based textual similarity as an additional matching signal.

### 🌲 Machine Learning Prediction

A **Random Forest Regressor** combines multiple engineered features to predict the resume-job match score.

### 💡 AI Recommendations

The system provides actionable suggestions such as:

* Missing skills
* Experience gaps
* Education requirements
* Certification gaps
* Resume improvement recommendations

### 📑 ATS Report

The application can generate an ATS analysis report containing the candidate's results and matching information.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    Resume Upload    │
                    │      PDF / DOCX     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Extraction   │
                    │    PDF / DOCX       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   NLP Processing    │
                    │                     │
                    │ Skills              │
                    │ Education           │
                    │ Experience          │
                    │ Certifications      │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌─────────────────────────────────┐
             │      Feature Engineering        │
             │                                 │
             │ TF-IDF Similarity               │
             │ Semantic Similarity             │
             │ Skill Match                     │
             │ Education Match                 │
             │ Experience Match                │
             │ Certification Match             │
             │ Resume/Job Length               │
             │ Skill Counts                    │
             └───────────────┬─────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Random Forest     │
                  │      Regressor      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   ATS Match Score   │
                  └──────────┬──────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       Streamlit Dashboard    │
              │                              │
              │ Score                        │
              │ Skills                       │
              │ Experience                   │
              │ Education                    │
              │ Certifications               │
              │ Recommendations              │
              └──────────────────────────────┘
```

---

# 🤖 Machine Learning Pipeline

The project uses a hybrid NLP + machine learning approach.

## 1. TF-IDF

TF-IDF measures lexical similarity between resume and job-description text.

It helps identify direct textual overlap between the two documents.

---

## 2. Semantic Embeddings

Transformer-based sentence embeddings are generated for both:

```text
Resume
   ↓
Sentence Embedding

Job Description
   ↓
Sentence Embedding
```

Cosine similarity is then used to calculate semantic similarity.

This allows the system to recognize related concepts even when exact words differ.

---

## 3. Feature Engineering

The trained model uses **14 engineered features**:

| Feature                   |
| ------------------------- |
| TF-IDF similarity         |
| Semantic similarity       |
| Skill match score         |
| Education match score     |
| Experience match score    |
| Certification match score |
| Resume length             |
| Job length                |
| Resume word count         |
| Job word count            |
| Candidate skill count     |
| Required skill count      |
| Skill overlap count       |
| Education exact match     |

---

# 🌲 Random Forest Model

The final prediction model is a **Random Forest Regressor**.

The model was selected using cross-validation and hyperparameter tuning.

### Best Parameters

```text
n_estimators = 300
max_depth = None
min_samples_split = 2
min_samples_leaf = 2
```

The model was trained using:

```text
9544 training samples
14 engineered features
```

---

# 📈 Model Performance

The trained model produced the following evaluation results:

| Metric |  Score |
| ------ | -----: |
| MAE    | 0.0878 |
| RMSE   | 0.1157 |
| R²     | 0.5159 |

### Interpretation

The model achieves an **R² of approximately 0.516**, meaning the engineered features explain a substantial portion of the variation in the target match score.

The MAE of **0.0878** indicates an average absolute prediction error of approximately 0.088 on the normalized target scale.

---

# 🔍 Feature Importance

The most important features identified by the trained Random Forest model were:

| Feature               | Importance |
| --------------------- | ---------: |
| TF-IDF Similarity     |     0.1650 |
| Job Length            |     0.1562 |
| Semantic Similarity   |     0.1333 |
| Resume Length         |     0.1206 |
| Candidate Skill Count |     0.1014 |
| Required Skill Count  |     0.0945 |
| Resume Word Count     |     0.0771 |
| Job Word Count        |     0.0714 |
| Skill Match Score     |     0.0421 |
| Education Match Score |     0.0150 |

This indicates that textual similarity and resume/job characteristics play a significant role in the model's predictions.

---

# 📂 Project Structure

```text
Resume_AI_Research/
│
├── app/
│   │
│   ├── app.py
│   ├── dashboard.py
│   ├── dashboard_skills.py
│   ├── components.py
│   ├── styles.py
│   ├── charts.py
│   ├── utils.py
│   ├── config.py
│   ├── predictor.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   │
│   └── train/
│       └── train_model.py
│
├── models/
│   ├── best_resume_model.pkl
│   ├── scaler.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── ...
```

---

# ⚙️ Technologies Used

## Programming

* Python

## Machine Learning

* Scikit-learn
* Random Forest
* Feature Engineering

## NLP

* Sentence Transformers
* TF-IDF
* spaCy
* PhraseMatcher

## Document Processing

* pdfplumber
* python-docx

## Frontend

* Streamlit
* HTML/CSS

## Data Processing

* Pandas
* NumPy

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/dakshsaini2/Resume_AI-Research.git
```

Navigate into the project:

```bash
cd Resume_AI-Research
```

---

## 2. Create a virtual environment

Windows:

```powershell
python -m venv resume_ai
```

Activate it:

```powershell
resume_ai\Scripts\activate
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Running the Application

From the project root:

```powershell
python -m streamlit run app\app.py
```

The Streamlit application will open in your browser.

---

# 🧪 Training the Model

If you need to retrain the model:

```powershell
python app\train\train_model.py
```

The training process generates:

```text
models/
├── best_resume_model.pkl
├── scaler.pkl
└── tfidf_vectorizer.pkl
```

---

# 📊 Dataset

The training dataset contains:

```text
9544 samples
35 columns
```

The target variable is:

```text
matched_score
```

The observed target range during training was:

```text
Minimum: 0.00
Maximum: 0.97
```

The training pipeline converts the original resume/job information into the engineered feature matrix used by the machine learning model.

---

# 🎯 Example Workflow

```text
1. Upload Resume
       ↓
2. Paste Job Description
       ↓
3. Extract Resume Information
       ↓
4. Extract Job Requirements
       ↓
5. Calculate NLP Features
       ↓
6. Generate Semantic Embeddings
       ↓
7. Calculate TF-IDF Similarity
       ↓
8. Calculate Skill/Education/Experience Matches
       ↓
9. Random Forest Prediction
       ↓
10. Generate ATS Score
       ↓
11. Display Dashboard
       ↓
12. Generate Recommendations
```

---

# 💡 Example Output

```text
ATS Match Score: 78.8%

Education:
✓ Requirement satisfied

Experience:
⚠ Experience gap detected

Skills:
✓ Python
✓ SQL
✓ Machine Learning
✓ Git

Missing:
+ Docker
+ AWS

Recommendation:
Improve missing technical skills and gain
additional relevant experience.
```

---

# 🔐 Model and Application Notes

The application separates the prediction pipeline into multiple components:

```text
Resume Extraction
        ↓
NLP Processing
        ↓
Feature Engineering
        ↓
Saved ML Model
        ↓
Prediction
        ↓
Dashboard
```

This makes the system easier to maintain and allows individual components such as skill extraction and experience extraction to be improved independently.

---

# ⚠️ Limitations

The current system has several limitations:

* Resume parsing is primarily text-based.
* Complex graphical resumes may not extract perfectly.
* Skill detection depends on the available skill database and synonym mappings.
* Experience extraction may not capture every possible resume format.
* The training target (`matched_score`) is dataset-dependent.
* The Random Forest model does not understand resume content directly; it operates on engineered numerical features.
* The current model should be treated as a decision-support system rather than an autonomous hiring decision maker.

---

# 🔮 Future Improvements

Potential improvements include:

* [ ] Improve resume parsing using advanced document-layout models
* [ ] Expand the skill database
* [ ] Improve certification recognition
* [ ] Add multilingual resume support
* [ ] Add OCR for scanned resumes
* [ ] Experiment with XGBoost/LightGBM
* [ ] Improve target-label quality
* [ ] Add explainable AI for individual predictions
* [ ] Add recruiter authentication
* [ ] Add candidate comparison
* [ ] Add resume ranking
* [ ] Deploy the application publicly
* [ ] Add automated model monitoring

---

# 🌟 Key Highlights

This project demonstrates practical implementation of:

```text
✓ Natural Language Processing
✓ Semantic Search
✓ Sentence Embeddings
✓ TF-IDF
✓ Feature Engineering
✓ Random Forest Regression
✓ Resume Parsing
✓ Skill Extraction
✓ Experience Extraction
✓ Streamlit Dashboard Development
✓ ATS Scoring
✓ AI-based Recommendations
```

---

# 👨‍💻 Author

**Daksh Saini**

B.Tech — Computer Science / Engineering

GitHub:

https://github.com/dakshsaini2

---

# ⭐ Project Goal

The goal of ResumeAI is to build an intelligent and explainable resume screening system that combines traditional NLP techniques with semantic embeddings and machine learning to provide meaningful resume-job compatibility analysis.

If you find the project useful, consider giving the repository a ⭐.
