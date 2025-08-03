import torch
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# LSTMModel이라는 이름의 파일을 불러와서 모델 속 세부 설정들을 한 줄로 불러오기 위함. ( 코드 가독성을 위한 분리 작업임.)
from model.lstm_model import LSTMModel  # 필요 시

# 학습된 모델을 불러오기 위한 load_model 함수 선언.
def load_model(model_name="traffic_lstm_model", model_dir="./model"):
    # 1. 메타 정보 로드
    with open(f"{model_dir}/{model_name}_meta.json", "r") as f:
        meta = json.load(f)

    # 2. 모델 초기화
    model = LSTMModel(
        input_size=meta['input_size'],
        hidden_size=meta['hidden_size'],
        num_layers=meta['num_layers'],
        dropout=meta['dropout']
    )

    # 3. 가중치 로드
    model.load_state_dict(torch.load(f"{model_dir}/{model_name}.pth", map_location='cpu'))
    model.eval()

    # 4. 스케일러 로드
    with open(f"{model_dir}/{model_name}_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(f"{model_dir}/{model_name}_y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    print("✅ 모델 및 스케일러 로드 완료")
    return model, x_scaler, y_scaler




def evaluate_on_test_sample(model_name="traffic_lstm_model",
                            model_dir="./model",
                            test_input_path="test_inputs_sample.pkl",
                            test_output_path="test_outputs_sample.pkl"):
    """
    저장된 모델을 불러와 샘플 테스트 데이터로 성능을 평가합니다.
    """
    # 1. 모델 및 스케일러 로드
    model, x_scaler, y_scaler = load_model(model_name, model_dir)

    # 2. 테스트 데이터 로드
    with open(test_input_path, "rb") as f:
        X_test = pickle.load(f)  # shape: (N, 100, 27)

    with open(test_output_path, "rb") as f:
        y_test = pickle.load(f)  # shape: (N, 1) or (N,)

    # 3. 입력 정규화
    N, T, F = X_test.shape
    X_scaled = x_scaler.transform(X_test.reshape(-1, F)).reshape(N, T, F)

    # 4. Tensor 변환
    X_tensor = torch.FloatTensor(X_scaled)

    # 5. 추론
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).squeeze().numpy()

    # 6. 역정규화
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 7. 정답 데이터 정리
    y_true = y_test.reshape(-1)

    # 8. 평가 지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"📊 테스트셋 성능 ({test_input_path} 기준)")
    print(f" - MAE: {mae:.4f}")
    print(f" - MSE: {mse:.4f}")

    return mae, mse

evaluate_on_test_sample(
    model_name="traffic_lstm_model",
    model_dir="./model",
    test_input_path="./task1_data/test_inputs_sample.pkl",
    test_output_path="./task1_data/test_outputs_sample.pkl"
)
