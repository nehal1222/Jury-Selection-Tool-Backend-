import pandas as pd
import random
from faker import Faker

fake = Faker('en_IN')  

num_jurors = 2000
regions = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Other']
education_levels = ['High School', 'Graduate', 'Postgraduate']
professions = ['Student', 'Teacher', 'Engineer', 'Lawyer', 'Business', 'Other']

questions = [
    "Do you believe everyone deserves a fair trial?",
    "Have you ever been involved in a legal dispute?",
    "How do you feel about strict punishments for crimes?",
    "Do you think your personal experiences could affect your judgment?",
    "What is your opinion on civil lawsuits?",
    "Do you trust the legal system to be impartial?",
    "How comfortable are you making decisions that affect others?",
    "Have you had any training or work in law enforcement?",
    "Do you think age or background influences judgment?",
    "What factors would you consider most important when deciding a case?",
    "Can you briefly describe your family background?",
    "Have your early life experiences shaped your perspective on justice?",
    "Do you think your cultural background influences your judgment?",
    "Have any personal experiences made you more empathetic or critical in decision-making?",
    "Do you think socio-economic factors affect how you view cases?",
    "Are there any life experiences that could impact your decisions as a juror?"
]

responses = [
    "I strongly agree with this statement.",
    "I somewhat agree but with some reservations.",
    "I am neutral on this matter.",
    "I somewhat disagree with this viewpoint.",
    "I strongly disagree and have opposite experiences.",
    "I come from a middle-class family and it taught me fairness.",
    "My upbringing emphasized honesty and justice.",
    "I have experienced challenges that make me more empathetic.",
    "Cultural traditions have shaped how I view responsibility.",
    "My socio-economic background has made me sensitive to inequality.",
    "I try to remain unbiased despite personal life experiences."
]

jurors = []
for _ in range(num_jurors):
    juror_responses = [f"{q} {random.choice(responses)}" for q in questions]
    juror = {
        'Name': fake.name(),
        'Age': random.randint(25, 65),
        'Gender': random.choice(['Male', 'Female']),
        'Region': random.choice(regions),
        'Education': random.choice(education_levels),
        'Profession': random.choice(professions),
        'Political_Leaning': round(random.uniform(0, 10), 1),
        'Civil_Bias': round(random.uniform(0, 1), 2),
        'Criminal_Bias': round(random.uniform(0, 1), 2),
        'Fairness_Score': round(random.uniform(0, 1), 2),
        'Questionnaire': " | ".join(juror_responses),  # concatenate responses
        'Prosecution_Bias': round(random.uniform(0, 1), 2),  # <-- here
        'Defense_Bias': round(random.uniform(0, 1), 2),      # <-- here
        'Neutral_Bias': round(random.uniform(0, 1), 2),      # <-- here
        'Questionnaire': " | ".join(juror_responses)
    }
    jurors.append(juror)

df = pd.DataFrame(jurors)
df.to_csv('synthetic_jurors_realistic.csv', index=False)
print(df.head())
