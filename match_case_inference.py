import os
import re
import pickle
import json
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# [필수 라이브러리] StellarGraph
try:
    import stellargraph as sg
    from stellargraph.mapper import PaddedGraphGenerator
    from stellargraph.layer import GraphConvolution
    from stellargraph import StellarGraph
    print("[시스템] StellarGraph 로드 성공")
except ImportError:
    print("[치명적 오류] stellargraph 라이브러리가 없습니다. (pip install stellargraph)")
    sys.exit(1)

from hls_env import hls_env

# =========================================================
# [사용자 설정]
# =========================================================
CASE_IDS = [2]            
CASE_ROOT_DIR = "../CASE" 

MODE = "lut"              # "lut" or "dsp"
TARGET_VALUE = 5000       # [중요] 이 값과 가장 '유사한' 결과를 찾습니다.
TARGET_CP = 10.0          # CP 제한 (이건 넘으면 안 됨)

STOP_ON_EXACT_MATCH = True # True면 오차 0일 때 즉시 종료, False면 끝까지 탐색
TRIALS = 50               # 시도 횟수
TEMPERATURE = 3.0         # 탐험 강도

# [경로 설정]
RL_MODEL_DIR = "." 
GPP_MODEL_PATH = "../GPP"

# 1. RL 행동 모델 (Actor)
RL_MODEL_PATH = os.path.join(RL_MODEL_DIR, "ac_mu_5_rs_100_model.h5")

# 2. GPP 예측 모델 (Predictors)
MODELS_CONFIG = {
    "lut": {
        "embed": os.path.join(GPP_MODEL_PATH, "model_embedding_lut.h5"),
        "proxy": os.path.join(GPP_MODEL_PATH, "model_proxy_lut.h5")
    },
    "dsp": {
        "embed": os.path.join(GPP_MODEL_PATH, "model_embedding_dsp.h5"),
        "proxy": os.path.join(GPP_MODEL_PATH, "model_proxy_dsp.h5")
    },
    "cp": {
        "embed": os.path.join(GPP_MODEL_PATH, "model_embedding_cp.h5"),
        "proxy": os.path.join(GPP_MODEL_PATH, "model_proxy_cp.h5")
    }
}
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
    if not os.path.exists(dfg_path): return False

    with open(dfg_path, "r") as f: lines = f.readlines()
    
    node_name, node_features = [], []
    edge_source, edge_end = [], []
    node_dir, node_number_mapping = [], {}
    node_info_map = {}
    
    parent_edges, seen_nodes = [], set()
    current_section = None
    input_node = inter_node = out_node = node_count = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if "Inputs" in line: current_section = "INPUT"
            elif "Operations" in line: current_section = "OP"
            elif "Edges" in line: current_section = "EDGE"
            elif "Outputs" in line: current_section = "OUTPUT"
            continue
        parts = line.split()

        if current_section == "INPUT":
            n_name = parts[0]
            if n_name in seen_nodes: continue
            seen_nodes.add(n_name)
            
            precision = 32
            if len(parts) > 1 and "INT" in parts[1]:
                try: precision = int(re.findall(r'\d+', parts[1])[0])
                except: pass
            p_bin = to_binary(precision)
            
            feat = [1, 0, 0, 0, 0] + list(p_bin) + [0, 0]
            
            node_name.append(n_name)
            node_features.append(feat)
            node_number_mapping[node_count] = n_name
            node_info_map[node_count] = {'name': n_name, 'type': 'INPUT'}
            
            input_node += 1
            node_count += 1

        elif current_section == "OP":
            n_name = parts[0]
            if "[Parent:" in line:
                match = re.search(r'\[Parent:(\w+)\]', line)
                if match: parent_edges.append((n_name, match.group(1)))
            
            if n_name in seen_nodes: continue
            seen_nodes.add(n_name)
            
            precision = 32
            prec_str = [p for p in parts if "INT" in p]
            if prec_str:
                try: precision = int(re.findall(r'\d+', prec_str[0])[0])
                except: pass
            p_bin = to_binary(precision)

            if "LOOP" in line:
                feat = [0, 0, 0, 1, 0] + [0]*5 + [1, 0]
                node_dir.append(node_count)
                info_type = 'LOOP'
                op_type = None
            else:
                op_type_str = parts[1] if len(parts) > 1 else ""
                if '*' in op_type_str:
                    feat = [0, 0, 1, 0, 0] + list(p_bin) + [1, 0]
                    node_dir.append(node_count)
                    info_type = 'OP'; op_type = 'mul'
                elif '+' in op_type_str:
                    feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
                    info_type = 'OP'; op_type = 'add'
                else:
                    feat = [0, 1, 0, 0, 0] + list(p_bin) + [0, 0]
                    info_type = 'OP'; op_type = 'unknown'

            node_name.append(n_name)
            node_features.append(feat)
            node_number_mapping[node_count] = n_name
            node_info_map[node_count] = {'name': n_name, 'type': info_type, 'op': op_type}
            
            inter_node += 1
            node_count += 1

        elif current_section == "EDGE":
            if len(parts) >= 2:
                edge_source.append(parts[0])
                edge_end.append(parts[1])

        elif current_section == "OUTPUT":
            n_name = parts[0]
            if n_name in seen_nodes: continue
            seen_nodes.add(n_name)
            
            feat = [0, 0, 0, 0, 1] + [0]*5 + [0, 0]
            
            node_name.append(n_name)
            node_features.append(feat)
            node_number_mapping[node_count] = n_name
            node_info_map[node_count] = {'name': n_name, 'type': 'OUTPUT'}
            
            out_node += 1
            node_count += 1

    for child, parent in parent_edges:
        edge_source.extend([child, parent])
        edge_end.extend([parent, child])

    valid_set = set(node_name)
    final_s, final_t = [], []
    for s, t in zip(edge_source, edge_end):
        if s in valid_set and t in valid_set:
            final_s.append(s); final_t.append(t)

    edge_df = pd.DataFrame({'source': final_s, 'target': final_t})
    df_nodes = pd.DataFrame(node_features, index=node_name, columns=[f'f{i}' for i in range(12)])
    
    with open('info_edge', 'wb') as f: pickle.dump([edge_df], f)
    with open('info_plain_graph', 'wb') as f: pickle.dump([df_nodes], f)
    with open('multiplication_index', 'wb') as f: pickle.dump([node_dir], f)
    with open('node_mapping.pkl', 'wb') as f: pickle.dump(node_number_mapping, f)
    with open('node_info_map.pkl', 'wb') as f: pickle.dump(node_info_map, f)
    
    return True

def predict_with_gpp(gpp_models, actions, target_indices, mapping, base_nodes_df, edge_df):
    current_nodes_df = base_nodes_df.copy()
    
    for i, action in enumerate(actions):
        if i >= len(target_indices): break
        node_idx = target_indices[i]
        node_id = mapping.get(node_idx)
        
        if node_id in current_nodes_df.index:
            current_nodes_df.at[node_id, 'f11'] = int(action)

    try:
        sg_graph = StellarGraph(nodes=current_nodes_df, edges=edge_df)
    except Exception as e:
        print(f"[경고] 그래프 생성 중 에러: {e}")
        return {}

    generator = PaddedGraphGenerator(graphs=[sg_graph])
    inputs = generator.flow([sg_graph])

    preds = {}
    for key, (embed_m, proxy_m) in gpp_models.items():
        try:
            emb = embed_m.predict(inputs)
            val = proxy_m.predict(emb)
            preds[key] = float(np.abs(val[0][0]))
        except Exception as e:
            preds[key] = -1.0
            
    return preds

def generate_directives(actions, target_indices, mapping, top_func, node_info):
    directives_list = []
    for i, action in enumerate(actions):
        if i >= len(target_indices): break
        node_idx = target_indices[i]
        info = node_info.get(node_idx)
        if not info: continue
        
        node_name = info['name']
        info_type = info['type']
        op_type = info.get('op')
        
        if info_type == 'LOOP':
            if action == 1: cmd = f'set_directive_pipeline "{top_func}/{node_name}"'
            else: cmd = f'set_directive_pipeline -off "{top_func}/{node_name}"'
            directives_list.append(cmd)
        elif info_type == 'OP':
            if op_type == 'mul':
                # RL의 결정에 따라 구현 방식(Fabric/DSP) 지정
                # MODE에 상관없이 RL이 1이면 Fabric, 0이면 DSP (기존 로직 유지)
                if action == 1: cmd = f'set_directive_bind_op -op mul -impl fabric "{top_func}" {node_name}'
                else: cmd = f'set_directive_bind_op -op mul -impl dsp "{top_func}" {node_name}'
                directives_list.append(cmd)
            elif op_type in ['add', 'sub']:
                if action == 1: cmd = f'set_directive_bind_op -op {op_type} -impl fabric "{top_func}" {node_name}'
                else: cmd = f'# set_directive_bind_op -op {op_type} -impl fabric "{top_func}" {node_name}'
                directives_list.append(cmd)
    return directives_list

def main():
    print("\n" + "="*70)
    print(f"      HLS RL Optimizer [Target Similarity Mode]")
    print("="*70)
    print(f"[설정] Mode: {MODE}, Target Value: {TARGET_VALUE} (유사한 값 찾기)")

    custom_objects = {'GraphConvolution': GraphConvolution}

    if not os.path.exists(RL_MODEL_PATH):
        print(f"[오류] RL 모델 파일이 없습니다: {RL_MODEL_PATH}"); return
    rl_model = load_model(RL_MODEL_PATH, custom_objects=custom_objects, compile=False)
    
    gpp_models = {}
    print("[시스템] GPP 예측 모델 로딩...")
    for key, cfg in MODELS_CONFIG.items():
        pe = cfg['embed']
        pp = cfg['proxy']
        if os.path.exists(pe) and os.path.exists(pp):
            try:
                em = load_model(pe, custom_objects=custom_objects, compile=False)
                pm = load_model(pp, custom_objects=custom_objects, compile=False)
                gpp_models[key] = (em, pm)
                print(f"  - {key.upper()} GPP 로드 완료")
            except Exception as e:
                print(f"  - {key.upper()} 로드 실패: {e}")
        else:
            print(f"  - [경고] {key.upper()} 모델 경로 오류: {pe} 또는 {pp}")
    
    if not gpp_models:
        print("[경고] GPP 모델을 하나도 로드하지 못했습니다! 경로를 다시 확인해주세요.")

    for case_id in CASE_IDS:
        print(f"\n ▶ Processing CASE {case_id}")
        case_folder = os.path.join(CASE_ROOT_DIR, f'case_{case_id}')
        dfg_filename = os.path.join(case_folder, f'DFG_case_{case_id}.txt')
        top_func_name = f'case_{case_id}'

        if not preprocess_dfg(dfg_filename): continue
        
        with open('info_plain_graph', 'rb') as f: df_nodes = pickle.load(f)[0]
        with open('info_edge', 'rb') as f: edge_df = pickle.load(f)[0]
        with open('multiplication_index', 'rb') as f: target_indices = pickle.load(f)[0]
        with open('node_mapping.pkl', 'rb') as f: mapping = pickle.load(f)
        with open('node_info_map.pkl', 'rb') as f: node_info = pickle.load(f)

        try: env = hls_env(alpha=0.5, lambda0=0.5)
        except: continue

        best_result = None
        best_score = -float('inf') # 점수는 -(차이) 이므로 클수록 좋음 (0에 가까움)

        for trial in range(TRIALS):
            sys.stdout.write(f"\r    Trial {trial + 1} / {TRIALS} ...")
            sys.stdout.flush()

            state = env.reset(0, 999)
            done = False
            actions_taken = []
            
            env_lut, env_dsp, env_cp = 99999, 999, 999.0

            while not done:
                state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
                
                action_probs, _ = rl_model(state_tf)
                logits = np.log(np.squeeze(action_probs.numpy()) + 1e-10) / TEMPERATURE
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)
                action = np.random.choice(len(probs), p=probs)
                
                state, _, done, c_lut, c_dsp, c_cp = env.step(action)
                env_lut, env_dsp, env_cp = c_lut, c_dsp, c_cp
                actions_taken.append(int(action))

            # GPP 모델 예측
            preds = predict_with_gpp(gpp_models, actions_taken, target_indices, mapping, df_nodes, edge_df)
            
            lut = preds.get("lut") if preds.get("lut", -1) > 0 else env_lut
            dsp = preds.get("dsp") if preds.get("dsp", -1) > 0 else env_dsp
            cp  = preds.get("cp")  if preds.get("cp", -1) > 0 else env_cp

            # =========================================================
            # [수정됨] 타겟 유사도 기반 평가 로직
            # =========================================================
            current_diff = float('inf')
            is_valid = False

            # 1. CP 제약조건 확인 (Timing Violation 체크)
            if cp <= TARGET_CP:
                is_valid = True
                
                # 2. 선택 모드에 따른 차이 계산 (절대값)
                if MODE == 'lut':
                    current_diff = abs(lut - TARGET_VALUE)
                elif MODE == 'dsp':
                    current_diff = abs(dsp - TARGET_VALUE)
                elif MODE == 'cp':
                    current_diff = abs(cp - TARGET_VALUE)

            if is_valid:
                # 점수: 차이가 작을수록(0에 가까울수록) 높아야 하므로 음수로 변환
                current_score = -current_diff
                
                if best_result is None or current_score > best_score:
                    best_score = current_score
                    best_result = {"lut": lut, "dsp": dsp, "cp": cp, "actions": actions_taken}
                    print(f"\n    [★] 갱신! LUT={lut:.1f}, DSP={dsp:.1f} (Target Diff: {current_diff:.1f})")
                    
                    # 오차가 0이면 완벽한 매칭이므로 즉시 종료 (옵션)
                    if STOP_ON_EXACT_MATCH and current_diff == 0:
                        break 
        
        if best_result:
            tcl_directives = generate_directives(best_result['actions'], target_indices, mapping, top_func_name, node_info)
            final_json = {
                "solution_1": {
                    "directives": tcl_directives,
                    "LUT_op": [], "SLICE": 0,
                    "LUT": int(best_result['lut']), "FF": 0,
                    "DSP": int(best_result['dsp']), "CP": round(float(best_result['cp']), 4)
                }
            }
            save_name = f"solution_{MODE}_{TARGET_VALUE}.json"
            with open(os.path.join(case_folder, save_name), "w") as f:
                json.dump(final_json, f, indent=4, cls=NpEncoder)
            print(f"\n    [성공] 저장 완료: {save_name}")
        else:
            print(f"\n    [실패] 유효한 해(Timing 만족)를 찾지 못했습니다.")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
