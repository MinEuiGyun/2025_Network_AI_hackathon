import os

import joblib           # pip install joblib
import numpy as np                                  # pip install numpy
import pandas as pd                                 # pip install pandas
from neuralforecast.core import NeuralForecast      # pip install neuralforecast
from neuralforecast.models import NHITS

class NHITSModel:
    def __init__(self):
        # self.last_ds = None #학습 데이터의 마지막 시점.
        # self.training_model()
        # self.last_ds = pd.date_range('2024-01-01', periods=700000, freq='100ms').max()   # 데이터의 마지막 시점
        self.reasoning_model()

    def read_csv(self):
        # CSV 파일 경로 지정
        file_path = os.path.join(os.getcwd(), 'config', 'train_data.csv')
        # CSV 파일 읽기
        csv_df = pd.read_csv(file_path)
        return csv_df
    
    def training_model(self):
        csv_df = self.read_csv()
        # 1. 샘플 데이터 생성 (예시)
        # 100ms 간격으로 700000개 데이터 (약 100초 분량)
        n_samples = 700000
        df = pd.DataFrame({
            'unique_id': ['series_1'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='100ms'),
            'y': csv_df['traffic_volume']   # 임의의 traffic_volume 값
        })
        self.last_ds = df['ds'].max()   # 데이터의 마지막 시점

        # 2. 모델 및 NeuralForecast 객체 설정

        distributed_kwargs = dict(
            enable_progress_bar=True,   # 진행바 끄기
            logger=False,                # 로그 기록/파일화 완전 OFF
            enable_checkpointing=False,  # 체크포인트 저장도 끄기
        )
        HORIZON = 10           # 미래 1초, 100ms 간격 10개 시점 예측
        INPUT_SIZE = 100       # 최대 10초 과거 데이터 (100ms 간격 100개)
        model = NHITS(
            h=HORIZON,
            input_size=INPUT_SIZE,
            max_steps=30000,
            early_stop_patience_steps=20,
            random_seed=42,          # 추가: 시드 고정
            learning_rate=0.001,
            **distributed_kwargs        # 위 옵션들 적용
        )

        nf = NeuralForecast(models=[model], freq='100ms')

        # 3. 모델 학습
        nf.fit(df=df, val_size=35000)
        self.save_parameter(nf)

        # 4. 예측 수행 (최근 시점부터 향후 10개 시점 예측)
        forecasts = nf.predict(df=df)
        print(forecasts)

        # 5. 예측 결과 중 traffic_volume 최댓값 산출
        # forecasts 데이터프레임은 'unique_id', 'ds', 'NHITS' 컬럼 포함
        future_max = forecasts['NHITS'].max()

        print(f"향후 1초(10개 시점)의 traffic_volume 예측값 중 최댓값: {future_max}")

    def reasoning_model(self):
        nf = self.load_parameter()
        input_data = joblib.load("aif/data/test_inputs.pkl")
        print(input_data)
        max_i = input_data.shape[0]
        print(max_i)
        n = 100
        future_max_list = []
        for i in range(max_i):
            test_df = pd.DataFrame({
                'unique_id': ['test_series'] * n,
                'ds': pd.date_range('2025-01-01', periods=n, freq='100ms'),
                'y': input_data[:, :, 6][i]  # 예측하려는 과거 100개 값 (1D array or list)
            })
            # 4. 예측 수행 (최근 시점부터 향후 10개 시점 예측)
            forecasts = nf.predict(df=test_df)
            print(forecasts)

            # 5. 예측 결과 중 traffic_volume 최댓값 산출
            # forecasts 데이터프레임은 'unique_id', 'ds', 'NHITS' 컬럼 포함
            future_max = forecasts['NHITS'].max()
            future_max_list.append(future_max)

        # submission.pkl 파일로 저장
        future_max_array = np.array(future_max_list).reshape(-1, 1)  # shape: (163854,)
        print(future_max_array)
        joblib.dump(future_max_array, "submission.pkl")


    def save_parameter(self, nf):
        nf.save(
            path=os.path.join('model'),   # 모델 저장 경로
            model_index=None,                  # 저장할 모델 인덱스, 기본값 None(전체 모델)
            overwrite=True,                    # 동일 이름 파일 있을 시 덮어쓰기
            save_dataset=False                 # 데이터셋 저장 여부(모델만 저장시 False)
        )
    
    def load_parameter(self):
        nf = NeuralForecast.load(path='model')
        return nf

if __name__ == "__main__":
    test = NHITSModel()