import pandas as pd

df = pd.read_csv('diabetic_data_cleaned.csv')

# ── Add readable labels for dashboard ────────────────────

# Age group risk label
def age_risk(age):
    if age in ['[70-80)', '[80-90)']:
        return 'High Risk Age'
    elif age in ['[60-70)', '[90-100)', '[20-30)']:
        return 'Medium Risk Age'
    else:
        return 'Lower Risk Age'

df['age_risk_group'] = df['age'].apply(age_risk)

# Readmission label (human readable)
df['readmission_label'] = df['readmitted'].map({
    'NO'  : 'Not Readmitted',
    '>30' : 'Readmitted >30 Days',
    '<30' : 'Readmitted <30 Days (High Risk)'
})

# High risk flag as Yes/No
df['high_risk'] = df['readmitted_binary'].map({1: 'Yes', 0: 'No'})

# Hospital stay category
def stay_category(days):
    if days <= 2:
        return 'Short (1-2 days)'
    elif days <= 5:
        return 'Medium (3-5 days)'
    else:
        return 'Long (6+ days)'

df['stay_category'] = df['time_in_hospital'].apply(stay_category)

# Medication load category
def med_category(meds):
    if meds <= 10:
        return 'Low (≤10)'
    elif meds <= 20:
        return 'Medium (11-20)'
    else:
        return 'High (21+)'

df['medication_load'] = df['num_medications'].apply(med_category)

# ── Select final columns for Power BI ────────────────────
powerbi_cols = [
    'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
    'age_risk_group', 'time_in_hospital', 'stay_category',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'medication_load', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'insulin',
    'diabetesMed', 'change', 'admission_type_id',
    'discharge_disposition_id', 'readmitted',
    'readmission_label', 'readmitted_binary', 'high_risk'
]

df_powerbi = df[powerbi_cols]

# ── Save ──────────────────────────────────────────────────
df_powerbi.to_csv('healthcare_powerbi.csv', index=False)

print("Power BI export ready ✓")
print(f"Total records: {len(df_powerbi):,}")
print(f"Total columns: {len(df_powerbi.columns)}")
print("\nColumns exported:")
for col in df_powerbi.columns:
    print(f"  - {col}")