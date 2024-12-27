import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime

# Read and process the data
df = pd.read_csv("dataset4.csv")

# Create a figure with three subplots
plt.figure(figsize=(20, 6))
plt.suptitle('Traffic Volume Analysis', fontsize=16, y=0.98)

# 1. Average Volume by Hour
plt.subplot(131)
hours = range(0, 24)
hourly_volumes = df.groupby('HH')["Vol"].mean().tolist()

plt.bar(hours, hourly_volumes, color='skyblue', alpha=0.7)
plt.title('Average Traffic Volume by Hour')
plt.xlabel('Hour of Day (24-hour format)')
plt.ylabel('Average Volume')
plt.grid(axis='y', alpha=0.3)

# Add hour labels with proper formatting
plt.xticks(hours, [f'{hour:02d}:00' for hour in hours], rotation=45)

# 2. Average Volume by Weekday
plt.subplot(132)

# Convert dates to weekdays
df['weekday'] = pd.to_datetime(
    df['Yr'].astype(str) + '-' +
    df['M'].astype(str) + '-' +
    df['D'].astype(str)
).dt.day_name()

# Calculate average volume by weekday
groupedByWeekDay = df.groupby('weekday')["Vol"].mean()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
volumes_by_weekday = [groupedByWeekDay[day] for day in weekdays]

bars = plt.bar(weekdays, volumes_by_weekday, color='lightgreen', alpha=0.7)
plt.title('Average Traffic Volume by Weekday')
plt.xlabel('Day of Week')
plt.ylabel('Average Volume')
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom')

# 3. Average Volume by Day of Year
plt.subplot(133)

# Create a dictionary for cumulative days in year
dict_cumulative_days = {
    0: 0, 1: 31, 2: 59, 3: 90, 4: 120, 5: 151,
    6: 181, 7: 212, 8: 243, 9: 273, 10: 304, 11: 334
}

# Calculate day of year
df["Day"] = [dict_cumulative_days[month - 1] + day
             for month, day in zip(df["M"].tolist(), df["D"].tolist())]

# Calculate average volume by day
days = df["Day"].unique().tolist()
volumes_by_day = df.groupby("Day")["Vol"].mean().tolist()

plt.scatter(days, volumes_by_day, alpha=0.5, color='coral', s=30)
plt.title('Average Traffic Volume by Day of Year')
plt.xlabel('Day of Year')
plt.ylabel('Average Volume')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(days, volumes_by_day, 1)
p = np.poly1d(z)
plt.plot(days, p(days), "r--", alpha=0.8, label='Trend line')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Find peak hour and format it properly
peak_hour = hours[hourly_volumes.index(max(hourly_volumes))]
peak_volume = max(hourly_volumes)

# Find busiest weekday
busiest_weekday_idx = volumes_by_weekday.index(max(volumes_by_weekday))
busiest_weekday = weekdays[busiest_weekday_idx]

# Print summary statistics
print("\nTraffic Volume Analysis Summary:")
print("-" * 50)
print(f"Total days analyzed: {len(days)}")
print(f"Peak hour volume: {peak_volume:.0f} (Hour {peak_hour:02d}:00)")
print(f"Busiest weekday: {busiest_weekday}")
print(f"Average daily volume: {sum(volumes_by_day)/len(volumes_by_day):.0f}")



def process_geo_clusters(df, threshold=3):
    """
    Process geographical data by grouping coordinates, counting occurrences,
    removing outlier groups, and maintaining original dataframe size.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'latitude' and 'longitude' columns
    threshold (float): Z-score threshold for outlier detection (default=3)

    Returns:
    pandas.DataFrame: Processed DataFrame with outliers removed
    """
    # Create a copy of the original dataframe
    df_copy = df.copy()

    # Group by lat/long and get counts
    grouped = df_copy.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')

    # Calculate z-scores for the counts
    grouped['z_score'] = np.abs(stats.zscore(grouped['count']))

    # Identify non-outlier groups
    valid_groups = grouped[grouped['z_score'] <= threshold]

    # Create a boolean mask for rows to keep
    mask = pd.merge(
        df_copy,
        valid_groups[['Latitude', 'Longitude']],
        on=['Latitude', 'Longitude'],
        how='left'
    ).notna().iloc[:, 0]

    # Apply mask to original dataframe
    df_processed = df_copy[mask]

    # If rows were removed, randomly sample from valid points to maintain size
    if len(df_processed) < len(df):
        rows_needed = len(df) - len(df_processed)
        extra_rows = df_processed.sample(n=rows_needed, replace=True)
        df_processed = pd.concat([df_processed, extra_rows], ignore_index=True)

    return df_processed

# Read and process data
df = pd.read_csv("collision_data_2017.csv")
df.dropna(subset=['Latitude', 'Longitude'])
df_processed = process_geo_clusters(df)
df_processed["Weekday"] = pd.to_datetime(df_processed['Date']).dt.day_name()

# Set style parameters manually
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Create figure with subplots
fig = plt.figure(figsize=(20, 10))
fig.suptitle('NYC Traffic Collision Analysis 2017', fontsize=16, y=0.98)

# 1. Daily Crashes Plot
ax1 = plt.subplot(131)
dfc = df_processed.groupby("Date").size().reset_index(name='count')
ax1.scatter(range(1, len(dfc) + 1), dfc["count"], alpha=0.5, color='#1f77b4', s=30)
ax1.set_title('Daily Crash Distribution', pad=15)
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Number of Crashes')

# Add trend line
z = np.polyfit(range(1, len(dfc) + 1), dfc["count"], 1)
p = np.poly1d(z)
ax1.plot(range(1, len(dfc) + 1), p(range(1, len(dfc) + 1)), "r--", alpha=0.8,
         label=f'Trend line')
ax1.legend()

# 2. Crashes by Weekday Plot
ax2 = plt.subplot(132)
dfw = df_processed.groupby("Weekday").size().reset_index(name='count')
# Reorder weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dfw['Weekday'] = pd.Categorical(dfw['Weekday'], categories=weekday_order, ordered=True)
dfw = dfw.sort_values('Weekday')

bars = ax2.bar(dfw['Weekday'], dfw['count'], color='#2ecc71')
ax2.set_title('Crashes by Weekday', pad=15)
ax2.set_xlabel('Weekday')
ax2.set_ylabel('Number of Crashes')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom')

# 3. Hourly Distribution Plot
ax3 = plt.subplot(133)
dfh = df.copy()
dfh["Time"] = [x[:2] for x in dfh["Time"].tolist()]
dfh_grouped = dfh.groupby("Time").size().reset_index(name='count')

# Convert time strings to integers for proper ordering
dfh_grouped["Time"] = dfh_grouped["Time"].astype(int)
dfh_grouped = dfh_grouped.sort_values("Time")

ax3.plot(dfh_grouped["Time"], dfh_grouped["count"],
         marker='o', linewidth=2, color='#e74c3c', alpha=0.7)
ax3.set_title('Crashes by Hour of Day', pad=15)
ax3.set_xlabel('Hour (24-hour format)')
ax3.set_ylabel('Number of Crashes')
ax3.set_xticks(range(0, 24))
ax3.set_xticklabels([f'{x:02d}:00' for x in range(24)], rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# Add a bit more space at the top for the main title
plt.subplots_adjust(top=0.9)

plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)
print(f"Total number of crashes: {len(df_processed):,}")
print(f"Average daily crashes: {dfc['count'].mean():.1f}")
print(f"Peak hour for crashes: {dfh_grouped.loc[dfh_grouped['count'].idxmax(), 'Time']:02d}:00")
print(f"Most crashes on: {dfw.loc[dfw['count'].idxmax(), 'Weekday']}")
