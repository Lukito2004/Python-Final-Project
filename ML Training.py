import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import joblib
from torch.amp import autocast, GradScaler

# Enable tensor cores if available
torch.backends.cudnn.benchmark = True

# Create directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Read data more efficiently with specific dtypes
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

# Create datetime using dictionary format
df['DateTime'] = pd.to_datetime({
    'year': df['Yr'],
    'month': df['M'],
    'day': df['D'],
    'hour': df['HH'],
    'minute': df['MM']
})

# Vectorized feature engineering
df['Hour'] = df['DateTime'].dt.hour.astype('int8')
df['DayOfWeek'] = df['DateTime'].dt.dayofweek.astype('int8')
df['Month'] = df['DateTime'].dt.month.astype('int8')
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype('int8')
df['IsRushHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18]).astype('int8')
df['IsMorningRush'] = df['Hour'].isin([7, 8, 9]).astype('int8')
df['IsEveningRush'] = df['Hour'].isin([16, 17, 18]).astype('int8')
df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3]).astype('int8')

# Efficient encoding
le_direction = LabelEncoder()
df['Direction_encoded'] = le_direction.fit_transform(df['Direction']).astype('int8')

# Select features and target
features = [
    'Hour', 'DayOfWeek', 'Month', 'Direction_encoded', 'SegmentID',
    'IsWeekend', 'IsRushHour', 'IsMorningRush', 'IsEveningRush', 'Season'
]
X = df[features].values
y = df['Vol'].values

# Efficient scaling
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float32)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

# Save preprocessors
joblib.dump(le_direction, 'models/direction_encoder.joblib')
joblib.dump(scaler_X, 'models/scaler_X.joblib')
joblib.dump(scaler_y, 'models/scaler_y.joblib')

# Efficient data split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.2,
    random_state=42,
    stratify=df['IsRushHour']
)


class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Larger batch size for better GPU utilization
train_dataset = TrafficDataset(X_train, y_train)
test_dataset = TrafficDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True, num_workers=4)


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


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrafficModel(len(features)).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

# Training loop with mixed precision
num_epochs = 60
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Fast validation
    model.eval()
    val_loss = 0
    with torch.no_grad(), autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            val_loss += criterion(outputs, batch_y).item()

    avg_val_loss = val_loss / len(test_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'checkpoints/best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

# Load best model and create prediction function
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])


@torch.inference_mode()
def predict_volume(hour, day_of_week, month, direction, segment_id):
    # Load preprocessors
    le_direction = joblib.load('models/direction_encoder.joblib')
    scaler_X = joblib.load('models/scaler_X.joblib')
    scaler_y = joblib.load('models/scaler_y.joblib')

    # Calculate features
    is_weekend = int(day_of_week in [5, 6])
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18])
    is_morning_rush = int(hour in [7, 8, 9])
    is_evening_rush = int(hour in [16, 17, 18])
    season = pd.cut([month], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])[0]
    direction_encoded = le_direction.transform([direction])[0]

    input_data = np.array([[
        hour, day_of_week, month, direction_encoded, segment_id,
        is_weekend, is_rush_hour, is_morning_rush, is_evening_rush, season
    ]], dtype=np.float32)

    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.from_numpy(input_scaled).to(device)

    with autocast():
        prediction = model(input_tensor)

    prediction_scaled = prediction.cpu().numpy()
    return float(scaler_y.inverse_transform(prediction_scaled)[0])