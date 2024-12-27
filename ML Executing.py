import torch
import joblib
import pandas as pd
import numpy as np
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.amp import autocast


class TrafficModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(num_samples=150, seed=42):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model and preprocessors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('checkpoints/best_model.pt')
    le_direction = joblib.load('models/direction_encoder.joblib')
    scaler_X = joblib.load('models/scaler_X.joblib')
    scaler_y = joblib.load('models/scaler_y.joblib')

    # Load data with efficient dtypes
    dtype_dict = {
        'Yr': 'int16',
        'M': 'int8',
        'D': 'int8',
        'HH': 'int8',
        'MM': 'int8',
        'Direction': 'category',
        'SegmentID': 'int16',
        'Vol': 'float32'
    }
    df = pd.read_csv('dataset.csv', dtype=dtype_dict)

    # Sample rows
    total_samples = len(df)
    sample_indices = np.random.choice(total_samples, num_samples, replace=False)
    df_sample = df.iloc[sample_indices].copy()

    # Create datetime and features
    df_sample['DateTime'] = pd.to_datetime({
        'year': df_sample['Yr'],
        'month': df_sample['M'],
        'day': df_sample['D'],
        'hour': df_sample['HH'],
        'minute': df_sample['MM']
    })

    # Feature engineering
    df_sample['Hour'] = df_sample['DateTime'].dt.hour.astype('int8')
    df_sample['DayOfWeek'] = df_sample['DateTime'].dt.dayofweek.astype('int8')
    df_sample['Month'] = df_sample['DateTime'].dt.month.astype('int8')
    df_sample['IsWeekend'] = (df_sample['DayOfWeek'] >= 5).astype('int8')
    df_sample['IsRushHour'] = df_sample['Hour'].isin([7, 8, 9, 16, 17, 18]).astype('int8')
    df_sample['IsMorningRush'] = df_sample['Hour'].isin([7, 8, 9]).astype('int8')
    df_sample['IsEveningRush'] = df_sample['Hour'].isin([16, 17, 18]).astype('int8')
    df_sample['Season'] = pd.cut(df_sample['Month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3]).astype('int8')
    df_sample['Direction_encoded'] = le_direction.transform(df_sample['Direction']).astype('int8')

    # Prepare features
    features = [
        'Hour', 'DayOfWeek', 'Month', 'Direction_encoded', 'SegmentID',
        'IsWeekend', 'IsRushHour', 'IsMorningRush', 'IsEveningRush', 'Season'
    ]
    X = df_sample[features].values
    y = df_sample['Vol'].values

    # Scale features
    X_scaled = scaler_X.transform(X)

    # Load model
    model = TrafficModel(len(features)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        input_tensor = torch.FloatTensor(X_scaled).to(device)
        predictions = model(input_tensor).cpu().numpy().flatten()

    # Inverse transform predictions
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("\nError Metrics:")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.4f}")

    # Show sample predictions
    print("\nSample Predictions (first 5):")
    for actual, pred in zip(y[:5], predictions[:5]):
        print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}, Error: {abs(actual - pred):.2f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'actuals': y,
        'predictions': predictions
    }


if __name__ == "__main__":
    evaluate_model()