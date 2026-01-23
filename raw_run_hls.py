import os
import subprocess
import xml.etree.ElementTree as ET
import json
import shutil

# =========================================================
# [사용자 설정] 환경에 맞게 수정하세요
# =========================================================
TOP_FUNC_NAME = "case_2"
SOURCE_FILE = "case_2.cc"
PROJECT_NAME = "hls_project"
SOLUTION_NAME = "solution_" + TOP_FUNC_NAME
TARGET_PART = "xcau25p-ffvb676-2-e"  # FPGA 보드명 (Zynq-7020 기준)
TARGET_CLOCK = 10.0              # 목표 클럭 (ns)

# Vitis HLS 실행 명령어 (환경변수에 등록되어 있어야 함)
# 만약 'vitis_hls' 명령어를 찾지 못하면 전체 경로를 적어주세요.
# 예: "/tools/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls" 
# =========================================================

def create_tcl_script(tcl_path):
    """Vitis HLS 제어용 TCL 스크립트 생성"""
    tcl_content = f"""
    open_project {PROJECT_NAME}
    set_top {TOP_FUNC_NAME}
    add_files {SOURCE_FILE}
    open_solution "{SOLUTION_NAME}" -flow_target vivado
    set_part {{{TARGET_PART}}}
    create_clock -period {TARGET_CLOCK} -name default
    
    # C Synthesis 실행
    csynth_design
    
    # 결과가 마음에 들면 RTL Export (필요시 주석 해제)
    # export_design -format ip_catalog
    
    exit
    """
    with open(tcl_path, "w") as f:
        f.write(tcl_content)
    print(f"[Info] TCL 스크립트 생성 완료: {tcl_path}")

def parse_xml_report(xml_path):
    """합성 결과 XML 파싱"""
    if not os.path.exists(xml_path):
        print(f"[Error] 리포트 파일을 찾을 수 없습니다: {xml_path}")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        "Target_CP": TARGET_CLOCK,
        "Estimated_CP": 0.0,
        "Latency_Avg": 0,
        "Resources": {
            "BRAM": 0,
            "DSP": 0,
            "FF": 0,
            "LUT": 0,
            "URAM": 0
        }
    }

    # 1. Timing 파싱
    try:
        est_cp = root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
        if est_cp is not None:
            result["Estimated_CP"] = float(est_cp.text)
    except: pass

    # 2. Latency 파싱
    try:
        lat_avg = root.find(".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")
        if lat_avg is not None:
            result["Latency_Avg"] = int(lat_avg.text)
    except: pass

    # 3. Resource 파싱
    try:
        res_node = root.find(".//AreaEstimates/Resources")
        if res_node is not None:
            for child in res_node:
                tag = child.tag.upper()
                val = int(child.text)
                if "BRAM" in tag: result["Resources"]["BRAM"] = val
                elif "DSP" in tag: result["Resources"]["DSP"] = val
                elif "FF" in tag: result["Resources"]["FF"] = val
                elif "LUT" in tag: result["Resources"]["LUT"] = val
                elif "URAM" in tag: result["Resources"]["URAM"] = val
    except: pass

    return result

def main():
    # 현재 작업 경로 확인
    current_dir = os.getcwd()
    print(f"[Start] 작업 경로: {current_dir}")

    # 1. TCL 파일 생성
    tcl_filename = "run_hls.tcl"
    create_tcl_script(tcl_filename)

    # 2. Vitis HLS 실행
    print(f"[Run] Vitis HLS 합성 시작... (시간이 조금 걸립니다)")
    try:
        # -f 옵션으로 TCL 실행
        subprocess.run(['vitis-run', '--mode', 'hls', '--tcl', tcl_filename], check=True)
    except FileNotFoundError:
        print("[Critical Error] 'vitis_hls' 명령어를 찾을 수 없습니다.")
        print("PATH에 Vitis HLS가 등록되어 있는지 확인하거나, 스크립트 상단 VITIS_CMD를 수정하세요.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[Error] 합성 중 오류 발생: {e}")
        return

    # 3. 결과 XML 파일 찾기
    # 경로: ./hls_project/solution1/syn/report/case_2_csynth.xml
    report_path = os.path.join(
        current_dir, 
        PROJECT_NAME, 
        SOLUTION_NAME, 
        "syn/report", 
        f"{TOP_FUNC_NAME}_csynth.xml"
    )

    print(f"[Info] 리포트 파싱 중: {report_path}")
    parsed_data = parse_xml_report(report_path)

    if parsed_data:
        # 4. JSON 저장
        save_filename = "synthesis_result.json"
        with open(save_filename, "w") as f:
            json.dump(parsed_data, f, indent=4)
        
        print("\n" + "="*40)
        print("           [합성 결과 요약]")
        print("="*40)
        print(json.dumps(parsed_data, indent=4))
        print("="*40)
        print(f"[Success] 결과 파일 저장됨: {os.path.join(current_dir, save_filename)}")
        
        # (선택) 임시 프로젝트 폴더 삭제하려면 아래 주석 해제
        # shutil.rmtree(PROJECT_NAME) 

if __name__ == "__main__":
    main()
import os
import subprocess
import xml.etree.ElementTree as ET
import json
import shutil

# =========================================================
# [사용자 설정] 환경에 맞게 수정하세요
# =========================================================
TOP_FUNC_NAME = "case_2"
SOURCE_FILE = "case_2.cc"
PROJECT_NAME = "hls_project"
SOLUTION_NAME = "solution_" + TOP_FUNC_NAME
TARGET_PART = "xcau25p-ffvb676-2-e"  # FPGA 보드명 (Zynq-7020 기준)
TARGET_CLOCK = 10.0              # 목표 클럭 (ns)

# Vitis HLS 실행 명령어 (환경변수에 등록되어 있어야 함)
# 만약 'vitis_hls' 명령어를 찾지 못하면 전체 경로를 적어주세요.
# 예: "/tools/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls"
VITIS_CMD = "vitis_hls" 
# =========================================================

def create_tcl_script(tcl_path):
    """Vitis HLS 제어용 TCL 스크립트 생성"""
    tcl_content = f"""
    open_project {PROJECT_NAME}
    set_top {TOP_FUNC_NAME}
    add_files {SOURCE_FILE}
    open_solution "{SOLUTION_NAME}" -flow_target vivado
    set_part {{{TARGET_PART}}}
    create_clock -period {TARGET_CLOCK} -name default
    
    # C Synthesis 실행
    csynth_design
    
    # 결과가 마음에 들면 RTL Export (필요시 주석 해제)
    # export_design -format ip_catalog
    
    exit
    """
    with open(tcl_path, "w") as f:
        f.write(tcl_content)
    print(f"[Info] TCL 스크립트 생성 완료: {tcl_path}")

def parse_xml_report(xml_path):
    """합성 결과 XML 파싱"""
    if not os.path.exists(xml_path):
        print(f"[Error] 리포트 파일을 찾을 수 없습니다: {xml_path}")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        "Target_CP": TARGET_CLOCK,
        "Estimated_CP": 0.0,
        "Latency_Avg": 0,
        "Resources": {
            "BRAM": 0,
            "DSP": 0,
            "FF": 0,
            "LUT": 0,
            "URAM": 0
        }
    }

    # 1. Timing 파싱
    try:
        est_cp = root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
        if est_cp is not None:
            result["Estimated_CP"] = float(est_cp.text)
    except: pass

    # 2. Latency 파싱
    try:
        lat_avg = root.find(".//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")
        if lat_avg is not None:
            result["Latency_Avg"] = int(lat_avg.text)
    except: pass

    # 3. Resource 파싱
    try:
        res_node = root.find(".//AreaEstimates/Resources")
        if res_node is not None:
            for child in res_node:
                tag = child.tag.upper()
                val = int(child.text)
                if "BRAM" in tag: result["Resources"]["BRAM"] = val
                elif "DSP" in tag: result["Resources"]["DSP"] = val
                elif "FF" in tag: result["Resources"]["FF"] = val
                elif "LUT" in tag: result["Resources"]["LUT"] = val
                elif "URAM" in tag: result["Resources"]["URAM"] = val
    except: pass

    return result

def main():
    # 현재 작업 경로 확인
    current_dir = os.getcwd()
    print(f"[Start] 작업 경로: {current_dir}")

    # 1. TCL 파일 생성
    tcl_filename = "run_hls.tcl"
    create_tcl_script(tcl_filename)

    # 2. Vitis HLS 실행
    print(f"[Run] Vitis HLS 합성 시작... (시간이 조금 걸립니다)")
    try:
        # -f 옵션으로 TCL 실행
        subprocess.run([VITIS_CMD, "-f", tcl_filename], check=True)
    except FileNotFoundError:
        print("[Critical Error] 'vitis_hls' 명령어를 찾을 수 없습니다.")
        print("PATH에 Vitis HLS가 등록되어 있는지 확인하거나, 스크립트 상단 VITIS_CMD를 수정하세요.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[Error] 합성 중 오류 발생: {e}")
        return

    # 3. 결과 XML 파일 찾기
    # 경로: ./hls_project/solution1/syn/report/case_2_csynth.xml
    report_path = os.path.join(
        current_dir, 
        PROJECT_NAME, 
        SOLUTION_NAME, 
        "syn/report", 
        f"{TOP_FUNC_NAME}_csynth.xml"
    )

    print(f"[Info] 리포트 파싱 중: {report_path}")
    parsed_data = parse_xml_report(report_path)

    if parsed_data:
        # 4. JSON 저장
        save_filename = "synthesis_result.json"
        with open(save_filename, "w") as f:
            json.dump(parsed_data, f, indent=4)
        
        print("\n" + "="*40)
        print("           [합성 결과 요약]")
        print("="*40)
        print(json.dumps(parsed_data, indent=4))
        print("="*40)
        print(f"[Success] 결과 파일 저장됨: {os.path.join(current_dir, save_filename)}")
        
        # (선택) 임시 프로젝트 폴더 삭제하려면 아래 주석 해제
        # shutil.rmtree(PROJECT_NAME) 

if __name__ == "__main__":
    main()
