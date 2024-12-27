import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('dataset4.csv')

# Step 2: Data Preprocessing
# Convert date and time columns to datetime
data['date_time'] = pd.to_datetime(data['Yr'].astype(str) + '-' +
                                  data['M'].astype(str) + '-' +
                                  data['D'].astype(str) + ' ' +
                                  data['HH'].astype(str) + ':' +
                                  data['MM'].astype(str))

# Create new features for time-based analysis
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month
data['hour'] = data['date_time'].dt.hour

# Drop unnecessary columns and rows with missing values
data = data.drop(columns=['RequestID', 'WktGeom'], errors='ignore')
data = data.dropna()

# Use LabelEncoder instead of one-hot encoding for SegmentID
le = LabelEncoder()
data['SegmentID_encoded'] = le.fit_transform(data['SegmentID'])

# Step 3: Prepare features and target variable
X = data[['day_of_week', 'month', 'hour', 'SegmentID_encoded']]
y = data['Vol']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a memory-efficient version of Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,  # Use all available cores
    max_depth=10,  # Limit tree depth
    min_samples_leaf=10  # Require at least 10 samples per leaf
)
rf_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse:.2f}')

# Step 7: Analyze seasonal variations
data['predicted_vol'] = rf_model.predict(X)

# Group by month and calculate average predicted traffic volume
monthly_variation = data.groupby('month')['predicted_vol'].mean().reset_index()

# Step 8: Plot seasonal variations
plt.figure(figsize=(10, 5))
plt.plot(monthly_variation['month'], monthly_variation['predicted_vol'], marker='o')
plt.title('Average Predicted Traffic Volume by Month')
plt.xlabel('Month')
plt.ylabel('Average Predicted Traffic Volume')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid()
plt.show()

# Print feature importances
feature_importance = pd.DataFrame({
    'feature': ['day_of_week', 'month', 'hour', 'segment_id'],
    'importance': rf_model.feature_importances_
})
print("\nFeature Importances:")
print(feature_importance.sort_values('importance', ascending=False))