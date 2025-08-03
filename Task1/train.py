import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel, TFTModel, TransformerModel
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json

# -------------------------------
# [GLOBAL HYPERPARAMETERS]
# -------------------------------
INPUT_CHUNK_LENGTH = 100
OUTPUT_CHUNK_LENGTH = 10
EPOCHS = 3
BATCH_SIZE = 32
RANDOM_STATE = 42
DROPOUT_TFT = 0.2
DROPOUT_TRANSFORMER = 0.1
# -------------------------------
# [DATA & PATH SETTING]
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

input_cols = [
    'fwd_pkt_count', 'bwd_pkt_count', 'fwd_tcp_pkt_count', 'bwd_tcp_pkt_count',
    'fwd_udp_pkt_count', 'bwd_udp_pkt_count', 'traffic_volume',
    'fwd_tcp_flags_cwr_count', 'bwd_tcp_flags_cwr_count', 'fwd_tcp_flags_ecn_count',
    'bwd_tcp_flags_ecn_count', 'fwd_tcp_flags_ack_count', 'bwd_tcp_flags_ack_count',
    'fwd_tcp_flags_push_count', 'bwd_tcp_flags_push_count', 'fwd_tcp_flags_reset_count',
    'bwd_tcp_flags_reset_count', 'fwd_tcp_flags_syn_count', 'bwd_tcp_flags_syn_count',
    'fwd_tcp_flags_fin_count', 'bwd_tcp_flags_fin_count', 'fwd_tcp_window_size_avg',
    'bwd_tcp_window_size_avg', 'fwd_tcp_window_size_max', 'bwd_tcp_window_size_max',
    'fwd_tcp_window_size_min', 'bwd_tcp_window_size_min'
]

print("[INFO] 데이터 로딩 중...")
df = pd.read_csv(DATA_PATH)
print("데이터 shape:", df.shape)

# [2] 전처리 (StandardScaler로 입력/타겟 스케일링)
print("[INFO] 데이터 스케일링...")
x_scaler = StandardScaler()
X = x_scaler.fit_transform(df[input_cols].values)  # (N, 27)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(df[['peak_volume']].values)  # (N, 1)

# [3] Darts TimeSeries 객체로 변환 (다변량 입력, 단변량 타겟)
print("[INFO] Darts TimeSeries 생성(다변량 입력)...")
times = pd.date_range('2023-01-01', periods=len(df), freq='s')
series_X = TimeSeries.from_times_and_values(times, X)           # [N, 27]
series_y = TimeSeries.from_times_and_values(times, y)           # [N, 1]

# [4] SOTA 시계열 모델 정의
models = {
    'nbeats': NBEATSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=RANDOM_STATE,
        save_checkpoints=True,
        force_reset=True
        # NBEATS는 dropout 파라미터 미지원!
    ),
    'tft': TFTModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=RANDOM_STATE,
        dropout=DROPOUT_TFT,                # Dropout 직접 조절 가능!
        force_reset=True
    ),
    'transformer': TransformerModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        random_state=RANDOM_STATE,
        dropout=DROPOUT_TRANSFORMER,         # Dropout 직접 조절 가능!
        force_reset=True
    ),
}

# [5] 각각 모델 학습 & 개별 저장 (가중치, 스케일러, 메타데이터)
for name, model in models.items():
    print(f"\n[TRAIN] {name.upper()} 모델 학습 시작")
    if name == 'nbeats':
        model.fit(series_y, epochs=EPOCHS, verbose=True)
    else:
        model.fit(series_y, past_covariates=series_X, epochs=EPOCHS, verbose=True)
    
    # --- 모델별로 별도 파일명으로 저장 ---
    model_path = f"{MODEL_DIR}/traffic_{name}_model.pth"
    x_scaler_path = f"{MODEL_DIR}/traffic_{name}_model_scaler.pkl"
    y_scaler_path = f"{MODEL_DIR}/traffic_{name}_model_y_scaler.pkl"
    meta_path = f"{MODEL_DIR}/traffic_{name}_model_meta.json"

    model.save(model_path)
    with open(x_scaler_path, "wb") as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_path, "wb") as f:
        pickle.dump(y_scaler, f)
    meta = {
        'input_cols': input_cols,
        'input_chunk_length': INPUT_CHUNK_LENGTH,
        'output_chunk_length': OUTPUT_CHUNK_LENGTH,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'random_state': RANDOM_STATE,
        'dropout_tft': DROPOUT_TFT if name == 'tft' else None,
        'dropout_transformer': DROPOUT_TRANSFORMER if name == 'transformer' else None,
        'model_name': name,
        'model_path': model_path,
        'x_scaler_path': x_scaler_path,
        'y_scaler_path': y_scaler_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {name.upper()} 모델/스케일러/메타데이터 저장 완료!")

print("\n=== [모든 모델 학습 및 저장 완료] ===")

# [6] 예측 및 앙상블 (전체 데이터 기준. 실제 추론에선 샘플 단위로 예측!)
preds = []
for name, model in models.items():
    if name == 'nbeats':
        pred = model.predict(n=OUTPUT_CHUNK_LENGTH)
    else:
        pred = model.predict(n=OUTPUT_CHUNK_LENGTH, past_covariates=series_X)
    preds.append(pred.values().squeeze())
    print(f"[{name.upper()}] 예측값:", pred.values().squeeze())

ensemble_pred = np.mean(np.stack(preds, axis=0), axis=0)
print("\n[앙상블 예측(평균)]:", ensemble_pred)

# [7] 예측값 역변환(실제 단위)
ensemble_pred_origin = y_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).squeeze()
print("\n[앙상블 예측(역변환)]:", ensemble_pred_origin)

print("\n=== [전체 파이프라인 완료!] ===")