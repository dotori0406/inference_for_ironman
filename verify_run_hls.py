import os
import subprocess
import xml.etree.ElementTree as ET
import json
import shutil
import time

# =========================================================
# [사용자 설정] 경로 수정 완료
# =========================================================
# 현재 위치(RLMD)를 기준으로 CASE 폴더 경로 설정
CASE_ID = "8"
CASE_ROOT_DIR = "../CASE"  # RLMD의 형제 폴더인 CASE를 가리킴
CASE_FOLDER = os.path.join(CASE_ROOT_DIR, f"case_{CASE_ID}")

TOP_FUNC_NAME = f"case_{CASE_ID}"    # case_2
SOURCE_FILE = os.path.join(CASE_FOLDER, f"case_{CASE_ID}.cc")  # ../CASE/case_2/case_2.cc
JSON_FILE = os.path.join(CASE_FOLDER, "solution_lut_8000.json") # ../CASE/case_2/solution_lut_5000.json

SOLUTION_FOLDER = "solution_case_8"  # Vitis 프로젝트 내 솔루션 이름
TARGET_PART = "xcau25p-ffvb676-2-e"  # FPGA 보드명
TARGET_CLOCK = 10.0                  # 목표 클럭

# 프로젝트 폴더는 현재 위치(RLMD)에 생성됩니다.
PROJECT_DIR = f"{TOP_FUNC_NAME}_syn"
# =========================================================

def create_tcl_script(tcl_path, directives):
    """지시어를 포함한 TCL 스크립트 생성"""
    
    # [수정] add_files에 경로가 포함된 SOURCE_FILE 변수 사용
    tcl_content = f"""
    open_project {PROJECT_DIR}
    set_top {TOP_FUNC_NAME}
    add_files {SOURCE_FILE}
    open_solution "{SOLUTION_FOLDER}" -flow_target vivado
    set_part {{{TARGET_PART}}}
    create_clock -period {TARGET_CLOCK} -name default
    """
    
    tcl_content += "\n    # --- Directives from JSON ---\n"
    for cmd in directives:
        # 함수명 치환 (gemm -> case_2)
        if "gemm" in cmd and TOP_FUNC_NAME != "gemm":
            cmd = cmd.replace("gemm", TOP_FUNC_NAME)
        tcl_content += f"    {cmd}\n"
        
    tcl_content += """
    # --- Run Synthesis ---
    csynth_design
    exit
    """
    
    with open(tcl_path, "w") as f:
        f.write(tcl_content)
    print(f"[Info] TCL 스크립트 생성 완료: {tcl_path}")

def parse_xml_report(xml_path):
    if not os.path.exists(xml_path): return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    res = {"LUT": 0, "DSP": 0, "FF": 0, "BRAM": 0, "CP": 0.0, "Latency": 0}
    try: res["CP"] = float(root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod").text)
    except: pass
    try: res["Latency"] = int(root.find(".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency").text)
    except: pass
    try:
        nodes = root.find(".//AreaEstimates/Resources")
        for node in nodes:
            tag = node.tag.upper()
            if tag in res: res[tag] = int(node.text)
    except: pass
    return res

def main():
    print("="*60)
    print(f"      [검증 시작] {TOP_FUNC_NAME}")
    print(f"      소스 위치: {SOURCE_FILE}")
    print(f"      JSON 위치: {JSON_FILE}")
    print("="*60)

    # 1. JSON 읽기
    if not os.path.exists(JSON_FILE):
        print(f"[Error] JSON 파일을 찾을 수 없습니다: {JSON_FILE}")
        print(" -> 경로가 맞는지 확인해주세요.")
        return

    with open(JSON_FILE, "r") as f:
        data = json.load(f)
        try:
            directives = data["solution_1"]["directives"]
        except KeyError:
            print("[Error] JSON 키 확인 필요 ('solution_1' -> 'directives')")
            return

    # 2. TCL 생성
    tcl_filename = "verify_hls.tcl"
    create_tcl_script(tcl_filename, directives)

    # 3. Vitis HLS 실행
    print(f"[Run] Vitis HLS 합성 시작 (vitis-run 사용)...")
    
    # (1) 기존 폴더 삭제
    if os.path.exists(PROJECT_DIR):
        subprocess.run(['rm', '-rf', PROJECT_DIR], check=True)

    # (2) vitis-run 실행
    try:
        subprocess.run(['vitis-run', '--mode', 'hls', '--tcl', tcl_filename], check=True)
    except FileNotFoundError:
        print("[Critical Error] 'vitis-run' 명령어를 찾을 수 없습니다. 환경 설정을 확인하세요.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[Error] 합성 중 에러 발생: {e}")
        return

    # 4. 결과 파싱
    report_path = f"{PROJECT_DIR}/{SOLUTION_FOLDER}/syn/report/{TOP_FUNC_NAME}_csynth.xml"
    
    if os.path.exists(report_path):
        result = parse_xml_report(report_path)
        print("\n" + "="*50)
        print(f"      [실제 검증 결과] {TOP_FUNC_NAME}")
        print("="*50)
        print(f"  - Target CP : {TARGET_CLOCK} ns")
        print(f"  - Est. CP   : {result['CP']} ns")
        print(f"  - Latency   : {result['Latency']} cycles")
        print("-" * 50)
        print(f"  - LUT       : {result['LUT']}")
        print(f"  - DSP       : {result['DSP']}")
        print(f"  - FF        : {result['FF']}")
        print("="*50)
        
        # 결과 저장 (RLMD 폴더에 저장됨)
        save_name = f"verification_result_{TOP_FUNC_NAME}.json"
        with open(save_name, "w") as f:
            json.dump(result, f, indent=4)
        print(f"[Info] 결과 저장 완료: {os.path.abspath(save_name)}")
    else:
        print(f"[Error] 리포트 파일 생성 실패.\n경로 확인: {report_path}")

if __name__ == "__main__":
    main()
