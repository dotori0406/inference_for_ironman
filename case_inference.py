import os
import re
import pickle
import json
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from hls_env import hls_env

# =========================================================
# [사용자 설정]
# =========================================================
CASE_IDS = [2]           # 실행할 케이스 번호
CASE_ROOT_DIR = "../CASE" # 데이터 위치 (현재 실행 위치 기준 상대경로)

MODE = "lut"             # "lut" or "dsp"
TARGET_VALUE = 5000       # 목표값
TARGET_CP = 10.0         # CP 제한

STOP_ON_SUCCESS = True   # 성공 시 중단
TRIALS = 50             # 시도 횟수
TEMPERATURE = 5.0        # 탐험 강도
MODEL_PATH = "ac_mu_5_rs_100_model.h5"
# =========================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def to_binary(number):
    return tuple((number >> i) & 1 for i in range(4, -1, -1))

def preprocess_dfg(dfg_path):
    print(f"    -> DFG 읽는 중: {dfg_path}")
    if not os.path.exists(dfg_path):
        print(f"    [오류] 파일을 찾을 수 없습니다: {dfg_path}")
        return False

    with open(dfg_path, "r") as f: lines = f.readlines()
    
    node_name, node_features, edge_source, edge_end = [], [], [], []
    node_dir, node_number_mapping = [], {}
    parent_edges, seen_nodes = [], set()
    current_section = None
    
    # 카운터 초기화
    input_node = inter_node = out_node = node_count = 0

    # 1차 파싱 (노드 및 엣지 수집)
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("#"):
            if "Inputs" in line: current_section = "INPUT"
            elif "Operations" in line: current_section = "OP"
            elif "Edges" in line: current_section = "EDGE"
            elif "Outputs" in line: current_section = "OUTPUT"
            continue
        parts = line.split()

        if current_section in ["INPUT", "OP", "OUTPUT"]:
            n_name = parts[0]
            if n_name in seen_nodes: continue
            
            # Precision 파싱
            precision = 32
            prec_str = [p for p in parts if "INT" in p]
            if prec_str:
                try: precision = int(re.findall(r'\d+', prec_str[0])[0])
                except: pass
            p_bin = to_binary(precision)
            
            feat = [0]*12
            is_target = False

            # 섹션별 피처 설정 (원본 로직 준수)
            if current_section == "INPUT":
                feat = [1, 0, 0, 0, 0] + list(p_bin) + [0, 0]
                input_node += 1
            elif current_section == "OUTPUT":
                feat = [0, 0, 0, 0, 1] + [0]*5 + [0, 0]
                out_node += 1
            elif current_section == "OP":
                if "[Parent:" in line:
                    match = re.search(r'\[Parent:(\w+)\]', line)
                    if match: parent_edges.append((n_name, match.group(1)))
                
                # LOOP 노드
                if "LOOP" in line:
                    feat = [0, 0, 0, 1, 0] + [0]*5 + [1, 0]
                    is_target = True # Loop도 지시어 대상
                else:
                    op_type = parts[1] if len(parts) > 1 else ""
                    if '*' in op_type:
                        feat = [0, 0, 1, 0, 0] + list(p_bin) + [1, 0] # Mul은 지시어 대상
                        is_target = True
                    elif '+' in op_type: 
                        feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
                    else: 
                        feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
                inter_node += 1
            
            seen_nodes.add(n_name)
            node_name.append(n_name)
            node_features.append(feat)
            node_number_mapping[node_count] = n_name
            if is_target: node_dir.append(node_count)
            node_count += 1

        elif current_section == "EDGE":
            if len(parts) >= 2:
                edge_source.append(parts[0])
                edge_end.append(parts[1])

    # 부모-자식 엣지 추가
    for child, parent in parent_edges:
        edge_source.extend([child, parent])
        edge_end.extend([parent, child])

    # =========================================================
    # [핵심 수정] 원본 preprocess 로직 복구: 유효하지 않은 엣지 제거
    # 이 부분이 없어서 m20 에러가 났던 것입니다.
    # =========================================================
    valid_nodes_set = set(node_name)
    final_sources = []
    final_targets = []
    
    for s, t in zip(edge_source, edge_end):
        if s in valid_nodes_set and t in valid_nodes_set:
            final_sources.append(s)
            final_targets.append(t)
        # else:
        #    print(f"    [Warning] 유령 엣지 제거됨: {s} -> {t}")
            
    edge_source = final_sources
    edge_end = final_targets
    # =========================================================

    edge_df = pd.DataFrame({'source': edge_source, 'target': edge_end})
    df_nodes = pd.DataFrame({'id': node_name})
    for i in range(12): df_nodes[f'f{i}'] = [nf[i] for nf in node_features]
    df_nodes = df_nodes.set_index("id")
    meta = [input_node, inter_node, out_node, len(node_dir), len(edge_source)]

    with open('info_edge', 'wb') as f: pickle.dump([edge_df], f)
    with open('info_plain_graph', 'wb') as f: pickle.dump([df_nodes], f)
    with open('info_meta', 'wb') as f: pickle.dump([meta], f)
    with open('multiplication_index', 'wb') as f: pickle.dump([node_dir], f)
    with open('node_mapping.pkl', 'wb') as f: pickle.dump(node_number_mapping, f)
    return True

def generate_directives(actions, target_indices, mapping, top_func):
    directives_list = []
    for i, action in enumerate(actions):
        if i >= len(target_indices): break
        node_idx = target_indices[i]
        node_name = mapping.get(node_idx, str(node_idx))
        
        if "L" in node_name:
            if action == 1: cmd = f'set_directive_pipeline "{top_func}/{node_name}"'
            else: cmd = f'set_directive_pipeline -off "{top_func}/{node_name}"'
            directives_list.append(cmd)
        elif "m" in node_name:
            if MODE == "lut":
                # LUT 모드: RL 1(Fabric) -> LUT 사용, RL 0(DSP) -> DSP 사용
                if action == 1: cmd = f'set_directive_bind_op -op mul -impl fabric "{top_func}" {node_name}'
                else: cmd = f'# set_directive_bind_op -op mul -impl dsp "{top_func}" {node_name}'
            else:
                if action == 1: cmd = f'set_directive_bind_op -op mul -impl fabric "{top_func}" {node_name}'
                else: cmd = f'# set_directive_bind_op -op mul -impl dsp "{top_func}" {node_name}'
            directives_list.append(cmd)
    return directives_list

def main():
    print("\n" + "="*70)
    print(f"      HLS RL Batch Optimizer (Fixed Ver.)")
    print("="*70)
    print(f"[설정] Mode: {MODE}, Target: {TARGET_VALUE}")

    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일({MODEL_PATH})이 없습니다.")
        return

    rl_model = load_model(MODEL_PATH, compile=False)
    
    for case_id in CASE_IDS:
        print("\n" + "-"*70)
        print(f" ▶ Processing CASE {case_id}")
        
        case_folder = os.path.join(CASE_ROOT_DIR, f'case_{case_id}')
        dfg_filename = os.path.join(case_folder, f'DFG_case_{case_id}.txt')
        top_func_name = f'case_{case_id}'

        # 전처리 실행 (안전장치 추가됨)
        if not preprocess_dfg(dfg_filename):
            continue

        try:
            env = hls_env(alpha=0.5, lambda0=0.5)
        except Exception as e:
            print(f"    [오류] 환경 초기화 실패: {e}")
            # StellarGraph 에러가 나면 다음 케이스로 넘어감
            continue

        env_dsp_target = TARGET_VALUE if MODE == 'dsp' else 999 
        
        best_result = None
        best_score = -float('inf')

        for trial in range(TRIALS):
            sys.stdout.write(f"\r    Running Trial {trial + 1} / {TRIALS} ...")
            sys.stdout.flush()

            state = env.reset(0, env_dsp_target)
            done = False
            actions_taken = []
            
            while not done:
                state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
                action_probs, _ = rl_model(state_tf)
                
                logits = np.log(np.squeeze(action_probs.numpy()) + 1e-10) / TEMPERATURE
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)
                
                action = np.random.choice(len(probs), p=probs)
                state, reward, done, lut, dsp, cp = env.step(action)
                actions_taken.append(int(action))

            is_success = False
            current_score = -float('inf')

            if cp <= TARGET_CP:
                if MODE == 'dsp':
                    if dsp <= TARGET_VALUE:
                        is_success = True; current_score = -lut
                elif MODE == 'lut':
                    if lut <= TARGET_VALUE:
                        is_success = True; current_score = -lut 
                elif MODE == 'cp':
                    if cp <= TARGET_VALUE:
                        is_success = True; current_score = -cp

            if is_success:
                if best_result is None or current_score > best_score:
                    best_score = current_score
                    best_result = {"lut": lut, "dsp": dsp, "cp": cp, "actions": actions_taken}
                    print(f"\n    [★] 갱신! LUT={lut}, DSP={dsp} (OK)")
                    
                    if STOP_ON_SUCCESS:
                        break 
        
        if best_result:
            with open('node_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
            with open('multiplication_index', 'rb') as f: target_indices = pickle.load(f)[0]
            
            tcl_directives = generate_directives(best_result['actions'], target_indices, mapping, top_func_name)
            
            final_json = {
                "solution_1": {
                    "directives": tcl_directives,
                    "LUT_op": [], "SLICE": 0,
                    "LUT": int(best_result['lut']), "FF": 0,
                    "DSP": int(best_result['dsp']), "CP": round(float(best_result['cp']), 4)
                }
            }
            
            save_name = f"solution_{MODE}_{TARGET_VALUE}.json"
            save_path = os.path.join(case_folder, save_name)
            
            with open(save_path, "w") as f:
                json.dump(final_json, f, indent=4, cls=NpEncoder)
            print(f"\n    [성공] 결과 저장 완료: {save_path}")
        else:
            print(f"\n    [실패] 목표 달성 실패.")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
