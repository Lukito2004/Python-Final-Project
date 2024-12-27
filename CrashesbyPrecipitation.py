import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('2017_NYC_Weather_Collision.csv')

# Define precipitation ranges and labels
precipitation_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, float('inf')]
precipitation_labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-1.0', '>1.0']

# Categorize the data into precipitation levels
data['Precipitation Level'] = pd.cut(data['precipitation (mm)'], bins=precipitation_bins, labels=precipitation_labels, right=False)

# Count the number of crashes per precipitation level
crash_counts = data['Precipitation Level'].value_counts().sort_index()

# Calculate percentages for labels
total_crashes = crash_counts.sum()
crash_percentages = (crash_counts / total_crashes * 100).round(1)

# Create figure with a reasonable size
plt.figure(figsize=(12, 8))

# Create pie chart
plt.pie(crash_counts.values,
        labels=[f'{label}\n({count} crashes, {pct}%)'
                for label, count, pct in zip(crash_counts.index,
                                           crash_counts.values,
                                           crash_percentages)],
        autopct='',  # Percentages are included in the labels
        startangle=90,
        counterclock=False,
        explode=[0.05] * len(crash_counts),
        textprops={'fontsize': 6})  # Smaller text for slices

# Add title
plt.title('Distribution of Crashes by Precipitation Level in NYC (2017)',
          pad=20,
          fontsize=14)

# Add legend
plt.legend(crash_counts.index,
          title="Precipitation Ranges",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of crashes: {total_crashes:,}")
print("\nCrashes by precipitation category:")
for category, count, percentage in zip(crash_counts.index,
                                     crash_counts.values,
                                     crash_percentages):
    print(f"{category}: {count:,} crashes ({percentage}%)")
