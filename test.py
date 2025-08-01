import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import ADIDA 

# 1. 과거 데이터를 준비 (최대 100개 시점; 최근 것이 마지막에 위치)
# 예시용 가상 데이터, 실제에선 관측값(real data) 사용
traffic_volume = np.random.randint(100, 500, size=100)
print(traffic_volume)
df = pd.DataFrame({
    'ds': pd.date_range("2025-08-01 01:00", periods=100, freq='s'),
    'y': traffic_volume
})
df['unique_id'] = 'series1'  # StatsForecast는 여러 시계열 지원, 단일이면 같은 값

# 2. 예측 모델 선정 (예시: AutoARIMA)
start_fit = time.time()
sf = StatsForecast(
    models=[ADIDA()],
    freq='s',   # 초 단위 시계열
    n_jobs=1
)

# 3. 모델 학습 및 예측 (10초 후까지 예측)
sf.fit(df)
end_fit = time.time()
print(f"학습 시간: {end_fit - start_fit:.4f} 초")
start_pred = time.time()
forecast_df = sf.predict(10)      # 10개 스텝(t+1 ~ t+10) 예측
end_pred = time.time()
print(f"예측 시간: {end_pred - start_pred:.4f} 초")
future_preds = forecast_df['ADIDA'].values

# 4. 예측값 중 최댓값 산출
max_pred = np.max(future_preds)
print("예측된 10초 내 최대 traffic_volume:", max_pred)

future_preds = forecast_df['ADIDA'].values
max_pred = np.max(future_preds)

# 그래프 시각화
plt.figure(figsize=(12, 5))
plt.plot(df['ds'], df['y'], label='past traffic_volume')
future_times = pd.date_range(df['ds'].iloc[-1] + pd.Timedelta(seconds=1), periods=10, freq='S')
plt.plot(future_times, future_preds, color='orange', marker='o', label='predict traffic_volume')
plt.axhline(max_pred, color='red', linestyle='--', label='predict value 10s in max')
plt.title('traffic_volume(과거+예측) 및 향후 10초간 최댓값')
plt.xlabel('time')
plt.ylabel('traffic_volume')
plt.legend()
plt.tight_layout()
plt.show()
