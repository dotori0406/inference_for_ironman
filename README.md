# inference_for_ironman
 <br>

 둘 모두 RLMD 폴더 안에서 hls_env.py 파일과 같은 위치에서 실행하면 정상적으로 사용 가능함 <br>
### case_inference.py 사용법 <br>
CASE_IDS = [2]           # 실행할 케이스 번호  
CASE_ROOT_DIR = "../CASE" # 데이터 위치 (현재 실행 위치 기준 상대경로)     <br>
   <br>
MODE = "lut"             # "lut" or "dsp" 모드 선택 가능 <br>
TARGET_VALUE = 5000       # 목표값 -> 어느정도 타당한 수치여야 가능함 <br>
TARGET_CP = 10.0         # CP 제한 -> 10이면 문제가 없는 듯 함 <br>
STOP_ON_SUCCESS = True   # 성공 시 중단 <br> 
TRIALS = 50             # 시도 횟수 -> epoch 같은.... <br>
TEMPERATURE = 5.0        # 탐험 강도 2~5 로 조절, 5가 탐험율이 높음 <br>
코드 내에서 원하는 대로 수정 필요 <br>

### inference.py 사용법 <br>
gemm_dfg에 맞춰 설정되어 있음 <br>
 <br>
MODE = "lut"           # 무조건 LUT 줄이기 <br>
TARGET_VALUE = 900     # 목표 LUT (이것만 넘기면 됨!) <br>
TARGET_CP = 10.0       # CP 제한 <br>
<br>
STOP_ON_SUCCESS = True # [핵심] True면 성공하자마자 멈춤 (False면 끝까지 더 좋은거 찾음) <br>

원하는 대로 수정하여 사용하면 됨 
