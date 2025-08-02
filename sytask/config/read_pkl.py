import json

import joblib           # pip install joblib
import numpy as np      # pip install numpy
import pandas as pd     # pip install pandas

class ReadPkl:
    def __init__(self):
        ######################################|| 개인 설정 ||####################################################################
        pkl_location = 'C:/CNU_Program/network/2025_Network_AI_hackathon/sytask/config/test_outputs_sample.pkl'     #.pkl 저장 위치 ※.pkl확장자까지 명시할 것!!!
        json_save_or_not = True                                                                                     # json 저장 여부, 만약 False일 경우 콘솔에만 출력
        json_location = 'C:/CNU_Program/network/2025_Network_AI_hackathon/sytask/config/output.json'                 # json 저장 위치 ※json 이름은 임의로 설정
        #########################################################################################################################
        data = self.read_pkl(pkl_location)
        if json_save_or_not:
            self.convert_json(data, json_location)
        
    def read_pkl(self, pkl_location):
        data = joblib.load(pkl_location)
        print(f"데이터 타입 : {type(data)}")
        print(f"데이터 내용 : {data}")
        return data
    
    def convert_json(self, data, json_location):
        if isinstance(data, dict):
            json_data = {k: self.convert_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            json_data = [self.convert_json(item) for item in data]
        elif isinstance(data, tuple):
            json_data = tuple(self.convert_json(item) for item in data)
        elif isinstance(data, np.ndarray):
            json_data = data.tolist()  # numpy 배열 → 리스트
        elif isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient='records')  # DataFrame → 리스트 딕셔너리 리스트 변환
        elif isinstance(data, pd.Series):
            json_data = data.tolist()
        # 필요한 타입들 추가 처리 가능
        else:
            # 기본 자료형, 혹은 str 변환 시도
            try:
                json.dumps(data)  # json화가 가능하면 그대로 반환
                json_data = data
            except (TypeError, OverflowError):
                json_data = str(data)  # 불가능하면 문자열로 변환

        # 파일 저장
        with open(json_location, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    read_pkl_class = ReadPkl()