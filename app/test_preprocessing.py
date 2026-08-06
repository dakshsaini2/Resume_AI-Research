from preprocessing import preprocess_resume

sample = """
Python Developer

Skills:
Python
SQL
Machine Learning
Docker

Experience:
2 years

Education:
Bachelor of Technology
"""

result = preprocess_resume(sample)

print(result)