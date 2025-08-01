import warnings
warnings.filterwarnings('ignore')

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# --- 설정 ---
SEQ_LEN = 3
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_CLASSES = 4
LEARNING_RATE = 0.001
EPOCHS = 1

# --- 1. 패킷 데이터에서 사용할 특성 정의 ---
PACKET_FEATURES = [
    'protocol', 'ip_len', 'ip_hdr_len', 'ip_tos', 'ip_ttl',
    'tcp_len', 'tcp_hdr_len', 'tcp_seq', 'tcp_ack',
    'tcp_window_size',
    'tcp_flags_ns', 'tcp_flags_cwr', 'tcp_flags_ecn',
    'tcp_flags_urg', 'tcp_flags_ack', 'tcp_flags_push',
    'tcp_flags_reset', 'tcp_flags_syn', 'tcp_flags_fin'
]

# --- 2. 사용자 정의 PyTorch 데이터셋 생성 ---
class FlowPacketDataset(Dataset):
    def __init__(self, flow_data, packet_data, seq_len=SEQ_LEN, features=PACKET_FEATURES):
        self.flow_data = flow_data
        self.packet_data = packet_data
        self.seq_len = seq_len
        self.features = features
        
    def __len__(self):
        return len(self.flow_data)
    
    def __getitem__(self, idx):
        packets_df = self.packet_data[idx]
        packets_df = packets_df[self.features]
        x = packets_df.head(self.seq_len)
        x_tensor = torch.FloatTensor(x.values.astype(np.float32))

        flow_row = self.flow_data.iloc[idx]
        y_duration = torch.LongTensor([int(flow_row['duration_class'])])
        y_volume = torch.LongTensor([int(flow_row['volume_class'])])
        
        return x_tensor, (y_duration, y_volume)

# --- 3. PyTorch Lightning LSTM 모델 정의 ---
class MultiTaskLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.head_duration = nn.Linear(hidden_size, num_classes)
        self.head_volume = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        logits_duration = self.head_duration(last_output)
        logits_volume = self.head_volume(last_output)
        return logits_duration, logits_volume
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_duration, y_volume = y
        logits_duration, logits_volume = self(x)
        
        loss_duration = F.cross_entropy(logits_duration, y_duration.squeeze())
        loss_volume = F.cross_entropy(logits_volume, y_volume.squeeze())
        total_loss = loss_duration + loss_volume
        
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("learning_rate", LEARNING_RATE))

# --- 4. 학습 및 저장 함수 ---
def train_and_save_model(model_name, save_dir):
    """
    모델을 학습하고, 추론에 필요한 파일들을 지정된 경로에 저장합니다.
    데이터는 현재 디렉토리에서 로드합니다.
    """
    # 저장 경로 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 로드 
    print("전처리된 데이터를 로드합니다...")

        # 경로를 현재 디렉토리 기준으로 수정
    train_flow_data = joblib.load("train_packet_data.pkl")
    train_packet_data = joblib.load("train_flow_data.pkl")
    train_flow_data = train_flow_data
    train_packet_data = train_packet_data

    # 데이터셋 및 데이터로더 생성
    print("데이터셋과 데이터로더를 생성합니다...")
    train_dataset = FlowPacketDataset(
        flow_data=train_flow_data,
        packet_data=train_packet_data,
        seq_len=SEQ_LEN,
        features=PACKET_FEATURES
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # 모델 초기화 및 학습
    print("모델을 초기화하고 학습을 시작합니다...")
    model = MultiTaskLSTM(
        input_size=len(PACKET_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    )
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", enable_progress_bar=True)
    trainer.fit(model, train_loader)
    
    print("\n--- 학습 완료 ---")

    # --- 추론 재현을 위한 파일 저장 ---
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    meta_path = os.path.join(save_dir, f'{model_name}_meta.json')
    
    # 1. 모델 가중치 저장
    torch.save(model.state_dict(), model_path)
    
    # 2. 메타데이터 생성 및 저장
    metadata = {
        'input_size': len(PACKET_FEATURES),
        'hidden_size': HIDDEN_SIZE,
        'num_classes': NUM_CLASSES,
        'seq_len': SEQ_LEN,
        'packet_features': PACKET_FEATURES
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"모델 및 메타데이터 저장 완료:")
    print(f"- 모델 가중치: {model_path}")
    print(f"- 메타데이터: {meta_path}")
    
# --- 5. 메인 실행 블록 ---
if __name__ == "__main__":
    # 모델과 메타데이터를 저장할 최종 디렉토리 (상대 경로로 수정)
    SAVE_DIRECTORY = "./submit/model"
    
    # 저장될 모델의 이름
    MODEL_NAME = "traffic_multitask_lstm"
    
    train_and_save_model(
        model_name=MODEL_NAME,
        save_dir=SAVE_DIRECTORY
    )