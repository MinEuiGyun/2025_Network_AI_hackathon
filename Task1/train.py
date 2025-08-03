import os
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# === 설정 ===
SEQ_LEN = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_CHANNELS = [64, 64, 64]
CSV_PATH = "./task1_data/train_data.csv"
SAVE_PATH = "./model"
MODEL_NAME = "traffic_tcn_manual"

# === Dataset 클래스 ===
class TrafficDataset(Dataset):
    def __init__(self, data, targets, seq_len=SEQ_LEN):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# === TCN 구조 ===
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out[:, :, :x.size(2)]  # skip connection shape match
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation,
                                        padding=(kernel_size - 1) * dilation,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y.transpose(1, 2)
        return self.linear(y[:, -1, :])

# === Lightning Module ===
class TCNRegressor(pl.LightningModule):
    def __init__(self, input_size, hidden_channels, lr, dropout):
        super().__init__()
        self.model = TCN(input_size, hidden_channels, dropout=dropout)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# === 학습 함수 ===
def train_model():
    # 데이터 로드
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 데이터 로드 완료: {df.shape}")

    input_cols = [col for col in df.columns if col != 'peak_volume']
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = x_scaler.fit_transform(df[input_cols].values)
    y = y_scaler.fit_transform(df[['peak_volume']].values).flatten()

    # train/val 분할
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 모델 정의
    model = TCNRegressor(
        input_size=len(input_cols),
        hidden_channels=HIDDEN_CHANNELS,
        lr=LEARNING_RATE,
        dropout=DROPOUT
    )

    # 학습
    trainer = pl.Trainer(max_epochs=3, accelerator='auto')
    trainer.fit(model, train_loader, val_loader)

    # 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_PATH}/{MODEL_NAME}.pth")
    with open(f"{SAVE_PATH}/{MODEL_NAME}_x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(f"{SAVE_PATH}/{MODEL_NAME}_y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open(f"{SAVE_PATH}/{MODEL_NAME}_meta.json", "w") as f:
        json.dump({
            "input_cols": input_cols,
            "seq_len": SEQ_LEN,
            "hidden_channels": HIDDEN_CHANNELS,
            "lr": LEARNING_RATE,
            "dropout": DROPOUT
        }, f, indent=4)

    print(f"✅ 모델 학습 및 저장 완료! → {SAVE_PATH}/{MODEL_NAME}.pth")

# === 실행 ===
if __name__ == "__main__":
    print("=== TCN 모델 수동 설정으로 학습 시작 ===")
    train_model()
