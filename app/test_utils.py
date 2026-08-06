from utils import *

resume = """
Python Developer

Skills:
Python
MySQL
GitHub
TensorFlow
Docker

Education:
Bachelor of Technology

Experience:
2 years

Certification:
AWS
"""

print("Skills:", extract_skills(resume))
print("Degree:", extract_degree(resume))
print("Experience:", extract_experience(resume))
print("Certifications:", extract_certifications(resume))
print("Summary:", generate_candidate_summary(resume))