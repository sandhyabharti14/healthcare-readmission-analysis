import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('diabetic_data_cleaned.csv')
os.makedirs('charts_pro', exist_ok=True)

# ── Global theme ──────────────────────────────────────────
BACKGROUND = '#F8F9FA'
PRIMARY     = '#2C3E50'
ACCENT      = '#E74C3C'
GREEN       = '#27AE60'
PURPLE      = '#8E44AD'
ORANGE      = '#E67E22'

def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, fontweight='bold', color=PRIMARY, pad=15)
    ax.set_xlabel(xlabel, fontsize=12, color=PRIMARY)
    ax.set_ylabel(ylabel, fontsize=12, color=PRIMARY)
    ax.tick_params(colors=PRIMARY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left','bottom']:
        ax.spines[spine].set_color('#CCCCCC')

# ── Chart 1: Readmission Distribution ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
order  = ['NO', '>30', '<30']
colors = [GREEN, ORANGE, ACCENT]
counts = [df[df['readmitted']==o].shape[0] for o in order]
bars   = ax.bar(order, counts, color=colors, edgecolor='white',
                linewidth=1.5, width=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 400,
            f'{count:,}', ha='center', fontsize=12,
            fontweight='bold', color=PRIMARY)
labels = ['Not Readmitted', 'Readmitted >30 Days', 'Readmitted <30 Days\n(High Risk)']
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
style_axis(ax, 'Patient Readmission Distribution',
           'Readmission Status', 'Number of Patients')
ax.set_ylim(0, max(counts)*1.15)
plt.tight_layout()
plt.savefig('charts_pro/01_readmission_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 1 saved ✓")

# ── Chart 2: Readmission Rate by Age ──────────────────────
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
             '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
age_rates = df.groupby('age')['readmitted_binary'].mean().reindex(age_order)*100
bar_colors = [ACCENT if r >= 9 else ORANGE if r >= 7 else GREEN
              for r in age_rates]
bars = ax.bar(age_order, age_rates, color=bar_colors,
              edgecolor='white', linewidth=1.5, width=0.6)
for bar, val in zip(bars, age_rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            f'{val:.1f}%', ha='center', fontsize=9,
            fontweight='bold', color=PRIMARY)
style_axis(ax, 'Early Readmission Rate by Age Group',
           'Age Group', 'Readmission Rate (%)')
ax.set_ylim(0, age_rates.max()*1.2)
plt.xticks(rotation=30)

# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT,  label='High Risk  ≥9%'),
                   Patch(facecolor=ORANGE, label='Medium Risk  7–9%'),
                   Patch(facecolor=GREEN,  label='Lower Risk  <7%')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('charts_pro/02_readmission_by_age.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 2 saved ✓")

# ── Chart 3: Time in Hospital vs Readmission ───────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
sns.boxplot(x='readmitted_binary', y='time_in_hospital',
            data=df, hue='readmitted_binary',
            palette={0: GREEN, 1: ACCENT},
            width=0.4, linewidth=1.5, legend=False,
            flierprops=dict(marker='o', markerfacecolor='#AAAAAA',
                            markersize=4, linestyle='none'),
            ax=ax)
ax.set_xticklabels(['Not Readmitted <30 Days', 'Readmitted <30 Days'],
                   fontsize=11)
style_axis(ax, 'Hospital Stay Duration vs Early Readmission',
           '', 'Days in Hospital')
plt.tight_layout()
plt.savefig('charts_pro/03_time_in_hospital.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 3 saved ✓")

# ── Chart 4: Medications vs Readmission ───────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
sns.boxplot(x='readmitted_binary', y='num_medications',
            data=df, hue='readmitted_binary',
            palette={0: GREEN, 1: ACCENT},
            width=0.4, linewidth=1.5, legend=False,
            flierprops=dict(marker='o', markerfacecolor='#AAAAAA',
                            markersize=4, linestyle='none'),
            ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Readmitted <30 Days', 'Readmitted <30 Days'],
                   fontsize=11)
style_axis(ax, 'Number of Medications vs Early Readmission',
           '', 'Number of Medications')
plt.tight_layout()
plt.savefig('charts_pro/04_medications.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 4 saved ✓")

# ── Chart 5: Insulin Usage vs Readmission ─────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BACKGROUND)
ax.set_facecolor(BACKGROUND)
insulin_rates = df.groupby('insulin')['readmitted_binary'].mean()*100
insulin_rates = insulin_rates.sort_values(ascending=False)
bar_colors = [ACCENT if v >= 10 else ORANGE if v >= 9 else GREEN
              for v in insulin_rates]
bars = ax.bar(insulin_rates.index, insulin_rates.values,
              color=bar_colors, edgecolor='white',
              linewidth=1.5, width=0.5)
for bar, val in zip(bars, insulin_rates):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{val:.1f}%', ha='center', fontsize=11,
            fontweight='bold', color=PRIMARY)
style_axis(ax, 'Early Readmission Rate by Insulin Usage',
           'Insulin Dosage Change', 'Readmission Rate (%)')
ax.set_ylim(0, insulin_rates.max()*1.2)
plt.tight_layout()
plt.savefig('charts_pro/05_insulin_usage.png', dpi=150,
            bbox_inches='tight', facecolor=BACKGROUND)
plt.close()
print("Chart 5 saved ✓")

print("\nAll professional charts saved in /charts_pro folder!")