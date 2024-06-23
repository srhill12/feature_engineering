

# Crowdfunding Campaign Analysis Using Feature Engineering

This project involves analyzing crowdfunding campaign data to predict the success of campaigns based on various features. The dataset includes details such as goal amount, pledged amount, backers count, and more.

## Dataset Information

- **Source**: Crowdfunding data from [BC-EDX](https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/crowdfunding-data.csv)
- **Instances**: 1682
- **Attributes**: goal, pledged, backers_count, country, staff_pick, spotlight, category, days_active, outcome, pledged_per_backer, backers_per_day, days_to_goal

## Steps

### Data Import and Feature Engineering

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import the data
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/datasets/crowdfunding-data.csv")

# Create a column "pledged_per_backer"
df['pledged_per_backer'] = df['pledged'] / df['backers_count']
df['pledged_per_backer'] = df['pledged_per_backer'].fillna(0)

# Create a backers_per_day column
df['backers_per_day'] = df['backers_count'] / df['days_active']

# Create a days_to_goal column
def days_to_goal(row):
    amount_remaining = row['goal'] - row['pledged']
    pledged_per_day = row['pledged_per_backer'] * row['backers_per_day']
    if pledged_per_day == 0:
        return 10000
    return amount_remaining / pledged_per_day

df['days_to_goal'] = df.apply(days_to_goal, axis=1)
