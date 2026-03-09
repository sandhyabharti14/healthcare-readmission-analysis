import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('diabetic_data_cleaned.csv')

# Create folder to save all charts
os.makedirs('charts', exist_ok=True)

sns.set_theme(style="whitegrid")
palette = ['#2ecc71', '#e74c3c']

# --- Chart 1: Readmission Distribution ---
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='readmitted', data=df, 
                   order=['NO', '>30', '<30'],
                   palette=['#3498db', '#f39c12', '#e74c3c'])
plt.title('Patient Readmission Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Readmission Status')
plt.ylabel('Number of Patients')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('charts/01_readmission_distribution.png', dpi=150)
plt.close()
print("Chart 1 saved.")

# --- Chart 2: Readmission by Age Group ---
plt.figure(figsize=(10, 5))
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_data = df.groupby('age')['readmitted_binary'].mean().reindex(age_order) * 100
age_data.plot(kind='bar', color='#e74c3c', edgecolor='black')
plt.title('Early Readmission Rate by Age Group (%)', fontsize=15, fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Readmission Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/02_readmission_by_age.png', dpi=150)
plt.close()
print("Chart 2 saved.")

# --- Chart 3: Time in Hospital vs Readmission ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='readmitted_binary', y='time_in_hospital', data=df,
            palette=palette)
plt.title('Time in Hospital vs Readmission', fontsize=15, fontweight='bold')
plt.xlabel('Readmitted Within 30 Days (0=No, 1=Yes)')
plt.ylabel('Days in Hospital')
plt.tight_layout()
plt.savefig('charts/03_time_in_hospital.png', dpi=150)
plt.close()
print("Chart 3 saved.")

# --- Chart 4: Number of Medications vs Readmission ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='readmitted_binary', y='num_medications', data=df,
            palette=palette)
plt.title('Number of Medications vs Readmission', fontsize=15, fontweight='bold')
plt.xlabel('Readmitted Within 30 Days (0=No, 1=Yes)')
plt.ylabel('Number of Medications')
plt.tight_layout()
plt.savefig('charts/04_medications.png', dpi=150)
plt.close()
print("Chart 4 saved.")

# --- Chart 5: Insulin Usage vs Readmission ---
plt.figure(figsize=(8, 5))
insulin_data = df.groupby('insulin')['readmitted_binary'].mean() * 100
insulin_data.sort_values(ascending=False).plot(kind='bar', 
                                                 color='#9b59b6', 
                                                 edgecolor='black')
plt.title('Readmission Rate by Insulin Usage (%)', fontsize=15, fontweight='bold')
plt.xlabel('Insulin Type')
plt.ylabel('Readmission Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/05_insulin_usage.png', dpi=150)
plt.close()
print("Chart 5 saved.")

print("\nAll EDA charts saved in /charts folder!")
print("\nKey Stats:")
print(f"Total Patients Analyzed: {len(df):,}")
print(f"High Risk Patients (<30 day readmission): {df['readmitted_binary'].sum():,}")
print(f"Overall Early Readmission Rate: {df['readmitted_binary'].mean()*100:.1f}%")