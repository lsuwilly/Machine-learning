import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
data = pd.read_csv('data/Crop_recommendation.csv')
print("Dataset Preview:")
print(data.head())

# columns = N,P,K,temperature,humidity,ph,rainfall,label
# units = mg/kg, mg/kg, mg/kg, °C, %, ph, mm, crop

# Create the plots directory if it doesn't exist
if not os.path.exists('./plots'):
    os.makedirs('./plots')

label_means = data.groupby('label').mean()

print(label_means)

units = {
    'N': 'mg/kg',
    'P': 'mg/kg',
    'K': 'mg/kg',
    'temperature': '°C',
    'humidity': '%',
    'ph': 'ph',
    'rainfall': 'mm'
}

for column in data.columns[:-1]:
    plt.figure(figsize=(12, 8))
    sorted_labels = label_means[column].sort_values().index
    data['label'] = pd.Categorical(
        data['label'], categories=sorted_labels, ordered=True)
    data.boxplot(column=column, by='label', grid=False)
    plt.title(f'{column} vs crop')
    plt.suptitle('')
    plt.xlabel('crop')
    plt.ylabel(f'{column} ({units[column]})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./plots/{column}_vs_crop.png', bbox_inches='tight', dpi=300)
    plt.close()
