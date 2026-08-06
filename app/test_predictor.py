
from predictor import predict_resume_score

resume = """
Python Developer

Skills:
Python
SQL
Machine Learning
TensorFlow
Docker

Experience:
3 years

Education:
B.Tech Computer Science

AWS Certified
"""

job = """
Python Developer Required

Skills

Python
SQL
Machine Learning
Docker

Experience
2 years

Education
B.Tech

AWS Certification
"""

result = predict_resume_score(
    resume,
    job
)

print("\n===== RESULT =====")
print("Score:", result["score"])
print("Recommendation:", result["recommendation"])

print("\n===== FEATURES =====")

for k, v in result["features"].items():
    print(f"{k:30} {v}")