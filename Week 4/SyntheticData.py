import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate random dates, with more realistic outliers
def random_date(start, end, outlier_chance=0.05):
    if random.random() < outlier_chance:
        if random.choice([True, False]):
            return start - timedelta(days=random.randint(1, 30))
        else:
            return end + timedelta(days=random.randint(1, 30))
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))

# Introducing correlations between symptoms
def generate_symptoms(symptoms):
    symptom_data = {}
    has_symptom = random.choice([True, False])

    for symptom in symptoms:
        if has_symptom:
            symptom_data[symptom] = random.choice([True] + [False]*2)
        else:
            symptom_data[symptom] = False
    return symptom_data

# Define the start and end date for the data
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)

# Define more complex geographic locations
locations = ['City_{}'.format(i) for i in range(1, 21)]

# Define symptoms
symptoms = ['High fever', 'Severe headache', 'Pain behind eyes', 'Joint pain', 'Rash']

# Generate synthetic data
data = []
for _ in range(1200):
    outlier_chance = 0.05
    missing_chance = 0.1

    patient_data = {
        'Date': random_date(start_date, end_date, outlier_chance=outlier_chance),
        'Location': random.choice(locations) if random.random() > missing_chance else None,
    }

    patient_data.update(generate_symptoms(symptoms))

    if random.random() < missing_chance:
        for symptom in symptoms:
            patient_data[symptom] = None
    
    print("Patient data: ", patient_data)
    data.append(patient_data)

# Create DataFrame
df = pd.DataFrame(data)

# Randomly drop some rows
df = df.drop(np.random.choice(df.index, size=int(len(df)*0.05), replace=False))

df.head()
