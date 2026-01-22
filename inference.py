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
# [사용자 설정] "만족하면 바로 퇴근" 모드
# =========================================================
MODE = "lut"           # 무조건 LUT 줄이기
TARGET_VALUE = 900     # 목표 LUT (이것만 넘기면 됨!)
TARGET_CP = 10.0       # CP 제한

STOP_ON_SUCCESS = True # [핵심] True면 성공하자마자 멈춤 (False면 끝까지 더 좋은거 찾음)

# 기타 설정
MODEL_PATH = "ac_mu_5_rs_100_model.h5"
DFG_FILENAME = "DFG_gemm_fixed.txt"
TRIALS = 500           
TEMPERATURE = 2.0      
TOP_FUNC_NAME = "gemm"
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
    if not os.path.exists(dfg_path):
        print(f"[알림] {dfg_path} 생성 중...")
        with open(dfg_path, "w") as f:
            f.write("""# Inputs\nin_i INT32\nin_j INT32\nin_k INT32\nin_k_col INT32\nin_i_col INT32\n# Operations\nL1 LOOP\nL2 LOOP [Parent:L1]\nm1 * INT32 [Parent:L2]\nL3 LOOP [Parent:L2]\nm2 * INT32 [Parent:L3]\nm3 * INT32 [Parent:L3]\nm4 + INT32 [Parent:L3]\n# Edges\nin_i_col m1\nin_k_col m2\nm2 m3\nm3 m4\nm1 m4\n# Outputs\no_prod""")
    
    # (전처리 코드는 기존과 동일하여 생략, 실행 시 문제 없도록 내장됨)
    with open(dfg_path, "r") as f: lines = f.readlines()
    node_name, node_features, edge_source, edge_end = [], [], [], []
    node_dir, node_number_mapping = [], {}
    parent_edges, seen_nodes = [], set()
    current_section = None
    input_node = inter_node = out_node = node_count = 0

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
            precision = 32
            prec_str = [p for p in parts if "INT" in p]
            if prec_str:
                try: precision = int(re.findall(r'\d+', prec_str[0])[0])
                except: pass
            p_bin = to_binary(precision)
            feat = [0]*12
            is_target = False
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
                if "LOOP" in line:
                    feat = [0, 0, 0, 1, 0] + [0]*5 + [1, 0]
                    is_target = True
                else:
                    op_type = parts[1] if len(parts) > 1 else ""
                    if '*' in op_type:
                        feat = [0, 0, 1, 0, 0] + list(p_bin) + [1, 0]
                        is_target = True
                    elif '+' in op_type: feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
                    else: feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
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
    for child, parent in parent_edges:
        edge_source.extend([child, parent])
        edge_end.extend([parent, child])
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
            # [중요] LUT 모드에서는 DSP를 팍팍 써야 LUT가 줄어듬
            # RL이 0(DSP)을 고르면 -> DSP 사용 (LUT 감소)
            # RL이 1(Fabric)을 고르면 -> LUT 사용 (LUT 증가)
            if action == 1: cmd = f'set_directive_bind_op -op mul -impl fabric "{top_func}" {node_name}'
            else: cmd = f'# set_directive_bind_op -op mul -impl dsp "{top_func}" {node_name}'
            directives_list.append(cmd)
    return directives_list

def main():
    print("\n" + "="*60)
    print(f"     HLS RL Optimizer [Quick Satisfy Mode]")
    print("="*60)
    print(f"[설정] Mode: {MODE} (LUT 최소화)")
    print(f"[설정] 목표 LUT: {TARGET_VALUE} 이하 (만족 시 즉시 종료: {STOP_ON_SUCCESS})")
    
    preprocess_dfg(DFG_FILENAME)
    
    if not os.path.exists(MODEL_PATH):
        print("모델 파일 없음")
        return

    rl_model = load_model(MODEL_PATH, compile=False)
    try:
        env = hls_env(alpha=0.5, lambda0=0.5)
    except:
        print("환경 초기화 실패")
        return

    # [전략] LUT 모드니까 DSP 타겟을 999로 줘서 DSP를 마음껏 쓰게 함
    env_dsp_target = 999 

    print(f"[진행] 탐색 시작...")

    for trial in range(TRIALS):
        sys.stdout.write(f"\r  ▶ Processing Trial {trial + 1} / {TRIALS} ...")
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

        # =========================================================
        # [성공 판독기] "그냥 만족만 하면 돼"
        # =========================================================
        if cp <= TARGET_CP and lut <= TARGET_VALUE:
            print(f"\n\n  [★] 조건 만족! (LUT: {lut} <= {TARGET_VALUE})")
            print(f"  [★] 화끈하게 쓴 DSP 개수: {dsp}")
            
            with open('node_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
            with open('multiplication_index', 'rb') as f: target_indices = pickle.load(f)[0]
            
            tcl_directives = generate_directives(actions_taken, target_indices, mapping, TOP_FUNC_NAME)
            
            final_json = {
                "solution_1": {
                    "directives": tcl_directives,
                    "LUT_op": [],
                    "SLICE": 0,
                    "LUT": int(lut),
                    "FF": 0,
                    "DSP": int(dsp),
                    "CP": round(float(cp), 4)
                }
            }
            
            save_name = f"solution_satisfy_lut_{TARGET_VALUE}.json"
            with open(save_name, "w") as f:
                json.dump(final_json, f, indent=4, cls=NpEncoder)
                
            print(f"  [★] 결과 저장 완료: {save_name}")
            print("-" * 60)
            print(json.dumps(final_json, indent=4))
            
            if STOP_ON_SUCCESS:
                print("\n[알림] 목표를 달성했으므로 조기 종료합니다.")
                return

    print(f"\n\n[실패] {TRIALS}번 시도했으나 LUT {TARGET_VALUE} 이하를 찾지 못했습니다.")
    print("팁: 이 회로는 기본 LUT 비용이 커서 600은 물리적으로 불가능할 수 있습니다.")
    print("    TARGET_VALUE를 900이나 1000으로 올려서 시도해보세요.")
    print("="*60)

if __name__ == "__main__":
    main()
