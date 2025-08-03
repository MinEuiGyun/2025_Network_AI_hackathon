import torch
import torch.nn as nn
import pytorch_lightning as pl

SEQ_LEN = 100  # 테스트 데이터와 맞춤
HIDDEN_SIZE = 128  # 더 긴 시퀀스를 위해 증가
BATCH_SIZE = 32

# 학습 당시의 모델의 설정들을 저장하기 위함 .

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
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)