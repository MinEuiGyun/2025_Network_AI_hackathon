import os
import numpy as np
import pandas as pd
import glob
import joblib
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
import json

if __name__ == '__main__':
    # 0. 재현성을 위한 시드 고정
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 디바이스 설정 (GPU 우선 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 테스트 데이터셋 클래스
    class TrafficTestDataset(Dataset):
        def __init__(self, x_data, seq_len=10):
            self.data = x_data
            self.seq_len = seq_len
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return torch.FloatTensor(self.data[idx])

    # 2. LSTM 모델 정의 
    class LSTMModel(pl.LightningModule):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
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
            last_output = lstm_out[:, -1, :]
            x = self.dropout(last_output)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
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

    # 3. 저장된 모델과 스케일러 로드
    def load_model_components(model_name="traffic_lstm_model", 
                            model_path="./model"):
        # 메타데이터 로드
        with open(f'{model_path}/{model_name}_meta.json', 'r') as f:
            metadata = json.load(f)
        
        # X 스케일러 로드
        with open(f'{model_path}/{model_name}_scaler.pkl', 'rb') as f:
            x_scaler = pickle.load(f)
        
        # Y 스케일러 로드 (있다면)
        y_scaler_path = f'{model_path}/{model_name}_y_scaler.pkl'
        
        with open(y_scaler_path, 'rb') as f:
            y_scaler = pickle.load(f)
        
        
        # 모델 로드
        input_size = metadata.get('input_size', 27)
        hidden_size = metadata.get('hidden_size', 64)
        num_layers = metadata.get('num_layers', 1)
        dropout = metadata.get('dropout', 0.2)
        
        model = LSTMModel(input_size, hidden_size, num_layers, dropout)
        model.load_state_dict(torch.load(f'{model_path}/{model_name}.pth', map_location=device))
        model.to(device)
        model.eval()
        
        return model, x_scaler, y_scaler, metadata

    # 4. 테스트 데이터 로드 및 전처리
    test_data = joblib.load("aif/data/test_inputs.pkl")

    # 모델 컴포넌트 로드
    model_name = "traffic_lstm_model"
    model, x_scaler, y_scaler, metadata = load_model_components(model_name)

    # 5. 테스트 데이터 전처리
    processed_test_data = []

    # test_data가 (163854, 100, 27) 형태의 numpy array인 경우
    if isinstance(test_data, np.ndarray) and len(test_data.shape) == 3:
        # 효율적인 방법: 전체를 한번에 처리
        n_samples, original_seq_len, n_features = test_data.shape
        
        # 2D로 reshape하여 스케일링
        test_data_2d = test_data.reshape(-1, n_features)
        test_data_scaled_2d = x_scaler.transform(test_data_2d)
        test_data_scaled = test_data_scaled_2d.reshape(n_samples, original_seq_len, n_features)
        
        # 시퀀스 길이 조정 (SEQ_LEN에 맞춤)
        seq_len = metadata.get('seq_len', 10)
        
        for i in range(n_samples):
            sample = test_data_scaled[i]  # (100, 27)
            
            if seq_len <= original_seq_len:
                # 마지막 seq_len개만 사용
                sequence = sample[-seq_len:]
            else:
                # 패딩이 필요한 경우 (거의 없을 것)
                pad_len = seq_len - original_seq_len
                padding = np.zeros((pad_len, n_features))
                sequence = np.vstack([padding, sample])
                
            processed_test_data.append(sequence)
            
    else:
        # test_data가 리스트 형태인 경우 (예전 방식)
        for sample in test_data:
            # 각 샘플을 개별적으로 처리
            sample_array = np.array(sample) if not isinstance(sample, np.ndarray) else sample
            
            # 스케일링 적용
            sample_scaled = x_scaler.transform(sample_array)
            
            # 시퀀스 길이 확인 및 조정
            seq_len = metadata.get('seq_len', 10)
            if len(sample_scaled) >= seq_len:
                sequence = sample_scaled[-seq_len:]
            else:
                pad_len = seq_len - len(sample_scaled)
                padding = np.zeros((pad_len, sample_scaled.shape[1]))
                sequence = np.vstack([padding, sample_scaled])
            
            processed_test_data.append(sequence)

    # numpy array로 변환
    processed_test_data = np.array(processed_test_data)

    # 6. 데이터셋 및 데이터로더 생성
    test_dataset = TrafficTestDataset(processed_test_data)
    BATCH_SIZE = 256
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 7. 추론 실행
    y_pred = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="[base_line]")
        for x_batch in test_pbar:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)  # (batch_size, 1)
            
            # peak_volume 예측값
            outputs_cpu = outputs.cpu().numpy().flatten()  # (batch_size,)
            
            # y_scaler가 있으면 역변환
            if y_scaler is not None:
                outputs_original = y_scaler.inverse_transform(outputs_cpu.reshape(-1, 1))
            else:
                outputs_original = outputs_cpu
            
            y_pred.append(outputs_original)

    # 8. 결과 합치기
    y_pred = np.concatenate(y_pred, axis=0)

    # 9. 결과 저장 (numpy array 형태)
    joblib.dump(y_pred, "submission.pkl")