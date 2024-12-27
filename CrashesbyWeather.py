import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('2017_NYC_Weather_Collision.csv')

# Count crashes for each weather type
weather_counts = df['Weather'].value_counts()

# Create the bar plot
plt.figure(figsize=(12, 6))
bars = sns.barplot(x=weather_counts.index, y=weather_counts.values)

# Customize the plot
plt.title('Number of Crashes by Weather Type in NYC (2017)', fontsize=14, pad=20)
plt.xlabel('Weather Type', fontsize=12)
plt.ylabel('Number of Crashes', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for i, v in enumerate(weather_counts.values):
    plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics
print("\nDetailed Weather Statistics:")
print("-" * 50)
print(f"Total number of crashes analyzed: {len(df):,}")
print("\nCrashes by weather type:")
for weather, count in weather_counts.items():
    percentage = round((count / len(df) * 100), 2)  # Fixed this line
    print(f"{weather}: {count:,} crashes ({percentage}%)")

# Optional: Display missing values if any
missing_weather = df['Weather'].isna().sum()
if missing_weather > 0:
    print(f"\nMissing weather data: {missing_weather:,} records")