import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os

# 설정
SEQ_LEN = 100  # 테스트 데이터와 맞춤
HIDDEN_SIZE = 128  # 더 긴 시퀀스를 위해 증가
BATCH_SIZE = 32

class TrafficDataset(Dataset):
    def __init__(self, data, targets, seq_len=SEQ_LEN):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # 시퀀스 (seq_len, features)
        y = self.targets[idx + self.seq_len]   # peak_volume 값
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 마지막 시퀀스의 출력 사용
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # (batch_size, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def train_model(model_name="traffic_lstm_model", 
                csv_path="train_data.csv ",
                save_path="./model"):
    os.makedirs(save_path, exist_ok=True)
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"데이터 로드 완료: {df.shape}")
    
    # 입력 특성 (27개, peak_volume 제외)
    input_cols = ['fwd_pkt_count', 'bwd_pkt_count', 'fwd_tcp_pkt_count', 'bwd_tcp_pkt_count',
                  'fwd_udp_pkt_count', 'bwd_udp_pkt_count', 'traffic_volume',
                  'fwd_tcp_flags_cwr_count', 'bwd_tcp_flags_cwr_count', 'fwd_tcp_flags_ecn_count',
                  'bwd_tcp_flags_ecn_count', 'fwd_tcp_flags_ack_count', 'bwd_tcp_flags_ack_count',
                  'fwd_tcp_flags_push_count', 'bwd_tcp_flags_push_count', 'fwd_tcp_flags_reset_count',
                  'bwd_tcp_flags_reset_count', 'fwd_tcp_flags_syn_count', 'bwd_tcp_flags_syn_count',
                  'fwd_tcp_flags_fin_count', 'bwd_tcp_flags_fin_count', 'fwd_tcp_window_size_avg',
                  'bwd_tcp_window_size_avg', 'fwd_tcp_window_size_max', 'bwd_tcp_window_size_max',
                  'fwd_tcp_window_size_min', 'bwd_tcp_window_size_min']
    
    # 데이터 전처리
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X = x_scaler.fit_transform(df[input_cols].values)  # 입력 특성
    y = y_scaler.fit_transform(df[['peak_volume']].values).flatten()  # 타겟
    
    # 데이터 분할
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 데이터셋 생성
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 모델 생성 및 학습
    model = LSTMModel(len(input_cols), hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.2)
    trainer = pl.Trainer(max_epochs=3, enable_progress_bar=True)
    trainer.fit(model, train_loader, val_loader)
    
    # 저장 경로 설정
    model_path = f'{save_path}/{model_name}.pth'
    x_scaler_path = f'{save_path}/{model_name}_scaler.pkl'
    y_scaler_path = f'{save_path}/{model_name}_y_scaler.pkl'
    meta_path = f'{save_path}/{model_name}_meta.json'
    
    # 저장
    torch.save(model.state_dict(), model_path)
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    
    metadata = {
        'cols': input_cols, 
        'seq_len': SEQ_LEN, 
        'hidden_size': HIDDEN_SIZE,
        'input_size': len(input_cols),
        'num_layers': 1,
        'dropout': 0.2
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"모델 저장 완료:")
    print(f"- 모델: {model_path}")
    print(f"- X 스케일러: {x_scaler_path}")
    print(f"- Y 스케일러: {y_scaler_path}")
    print(f"- 메타데이터: {meta_path}")
    return model, x_scaler, y_scaler

if __name__ == "__main__":
    # 학습 실행
    print("=== LSTM 모델 학습 시작 ===")
    model, x_scaler, y_scaler = train_model("traffic_lstm_model")
    print("=== 학습 완료! ===")