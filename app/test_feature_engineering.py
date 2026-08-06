from feature_engineering import generate_features

resume = """
Python Developer

Skills:
Python
SQL
Machine Learning
TensorFlow

Experience:
3 years

Education:
B.Tech Computer Science

AWS Certified
"""

job = """
Looking for Python Developer

Skills Required

Python
SQL
Machine Learning
Docker

Experience:
2 years

Education:
B.Tech

AWS certification preferred
"""

features = generate_features(
    resume,
    job
)

for k, v in features.items():
    print(f"{k:30} : {v}")