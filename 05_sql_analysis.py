import pandas as pd
import sqlite3
import os

df = pd.read_csv('diabetic_data_cleaned.csv')

# ── Create SQLite database ────────────────────────────────
conn = sqlite3.connect('healthcare.db')
df.to_sql('patients', conn, if_exists='replace', index=False)
print("Database created successfully ✓")
print(f"Total records loaded: {len(df):,}\n")

# ── Query 1: Overall Readmission Summary ──────────────────
q1 = """
SELECT 
    readmitted,
    COUNT(*) as total_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM patients
GROUP BY readmitted
ORDER BY total_patients DESC
"""
print("=" * 55)
print("QUERY 1: Overall Readmission Summary")
print("=" * 55)
print(pd.read_sql(q1, conn).to_string(index=False))

# ── Query 2: Top 5 High Risk Age Groups ───────────────────
q2 = """
SELECT 
    age,
    COUNT(*) as total_patients,
    SUM(readmitted_binary) as readmitted_count,
    ROUND(SUM(readmitted_binary) * 100.0 / COUNT(*), 2) as readmission_rate_pct
FROM patients
GROUP BY age
HAVING total_patients > 100
ORDER BY readmission_rate_pct DESC
LIMIT 5
"""
print("\n" + "=" * 55)
print("QUERY 2: Top 5 High Risk Age Groups")
print("=" * 55)
print(pd.read_sql(q2, conn).to_string(index=False))

# ── Query 3: Readmission by Number of Diagnoses ───────────
q3 = """
SELECT 
    number_diagnoses,
    COUNT(*) as total_patients,
    ROUND(SUM(readmitted_binary) * 100.0 / COUNT(*), 2) as readmission_rate_pct
FROM patients
GROUP BY number_diagnoses
ORDER BY number_diagnoses DESC
"""
print("\n" + "=" * 55)
print("QUERY 3: Readmission Rate by Number of Diagnoses")
print("=" * 55)
print(pd.read_sql(q3, conn).to_string(index=False))

# ── Query 4: Readmission by Race ──────────────────────────
q4 = """
SELECT 
    race,
    COUNT(*) as total_patients,
    SUM(readmitted_binary) as high_risk_count,
    ROUND(SUM(readmitted_binary) * 100.0 / COUNT(*), 2) as readmission_rate_pct
FROM patients
WHERE race IS NOT NULL
GROUP BY race
ORDER BY readmission_rate_pct DESC
"""
print("\n" + "=" * 55)
print("QUERY 4: Readmission Rate by Race")
print("=" * 55)
print(pd.read_sql(q4, conn).to_string(index=False))

# ── Query 5: Average Hospital Stay for High Risk Patients ─
q5 = """
SELECT 
    CASE WHEN readmitted_binary = 1 
         THEN 'High Risk (Readmitted <30d)' 
         ELSE 'Low Risk' END as patient_group,
    COUNT(*) as total_patients,
    ROUND(AVG(time_in_hospital), 2) as avg_days_in_hospital,
    ROUND(AVG(num_medications), 2) as avg_medications,
    ROUND(AVG(num_lab_procedures), 2) as avg_lab_procedures,
    ROUND(AVG(number_diagnoses), 2) as avg_diagnoses
FROM patients
GROUP BY readmitted_binary
"""
print("\n" + "=" * 55)
print("QUERY 5: High Risk vs Low Risk Patient Profile")
print("=" * 55)
print(pd.read_sql(q5, conn).to_string(index=False))

# ── Query 6: Most Common Admission Types for High Risk ────
q6 = """
SELECT 
    admission_type_id,
    COUNT(*) as total_patients,
    SUM(readmitted_binary) as high_risk_count,
    ROUND(SUM(readmitted_binary) * 100.0 / COUNT(*), 2) as readmission_rate_pct
FROM patients
GROUP BY admission_type_id
ORDER BY readmission_rate_pct DESC
"""
print("\n" + "=" * 55)
print("QUERY 6: Readmission Rate by Admission Type")
print("=" * 55)
print(pd.read_sql(q6, conn).to_string(index=False))

conn.close()
print("\nAll SQL queries executed successfully ✓")
print("Database saved as healthcare.db ✓")