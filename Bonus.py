import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('Housing.csv')

# Data preprocessing
print("Missing Values:\n", data.isnull().sum())
data.fillna(data.median(numeric_only=True), inplace=True)

# Define feature columns
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']


# Remove outliers using IQR method
def remove_outliers(df, columns):
    df_clean = df.copy()
    mask = pd.Series(True, index=df_clean.index)

    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = mask & (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)

    return df_clean[mask]


# Remove outliers from price and numerical columns
data = remove_outliers(data, ['price'] + numerical_cols)
print(f"\nRows after outlier removal: {len(data)}")

# Create correlation matrix
numerical_data = data[numerical_cols + ['price']]
correlation_matrix = numerical_data.corr()

# EDA
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Price vs Area scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='area', y='price')
plt.title('Price vs Area')
plt.show()

# Price distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['price'], kde=True, bins=30)
plt.title('Distribution of Price')
plt.show()

# Feature Engineering
data['price_per_sqft'] = data['price'] / data['area']

# Model Building
X = data.drop(['price', 'price_per_sqft'], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("\nModel Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R²: {r2_score(y_test, predictions):.3f}")

# Hyperparameter tuning
param_grid = {
    'regressor__fit_intercept': [True, False]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)

# Final evaluation
tuned_predictions = grid_search.predict(X_test)
print("\nTuned Model Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, tuned_predictions)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, tuned_predictions):.2f}")
print(f"R²: {r2_score(y_test, tuned_predictions):.3f}")

# Save model
import joblib

joblib.dump(grid_search.best_estimator_, 'house_price_model.pkl')