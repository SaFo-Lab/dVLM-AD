#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Waymo E2E planning:
1) ADE over validation frames
2) Rater Feedback Score (RFS)

Usage:
  python -m waymo.eval_planning \
    --dataset_folder /weka/home/xliu316/scratchcxiao13/yingzi/workspace/waymo \
    --val_glob "val*.tfrecord*" \
    --pred_json ./waymo/ad_finetune_val_cot_planning_30k_final.json \
    --ade_proposal best   # {first,argmax,best}

Requirements:
  - tensorflow>=2.x
  - waymo-open-dataset pip 包（含 metrics.python.rater_feedback_utils）
  - 你拥有 WOD-E2E 的 .tfrecord 验证集
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Waymo metrics: Rater Feedback Score
from .utils import get_rater_feedback_score  # noqa: E402

# ---- Waymo E2E protos（不同版本类型名可能不同，这里做兜底尝试）
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e_data_pb2  # noqa: E402
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from PIL import Image
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import os, json, numpy as np, tensorflow as tf
from tqdm import tqdm



def extract_waypoints_from_frame_bytes(frame_bytes):
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(frame_bytes)

    # print(data.past_states)
    # print(data.intent)
    # print(data.preference_trajectories)

    # 有的字段可能缺失，做个健壮判断
    px = list(data.future_states.pos_x) if len(data.future_states.pos_x) else []
    py = list(data.future_states.pos_y) if len(data.future_states.pos_y) else []
    pz = list(data.future_states.pos_z) if len(data.future_states.pos_z) else []

    n = min(len(px), len(py), len(pz))
    waypoints = [[float(px[i]), float(py[i])] for i in range(n)]

    # 可用作唯一键的标识：context.name + timestamp
    frame_id = data.frame.context.name if data.frame.context.name else ""
    ts = int(data.frame.timestamp_micros)

    image_list = [
        os.path.join(DATASET_FOLDER, split, frame_id.split("-")[0], frame_id.split("-")[-1] + "_CAM_FRONT_LEFT.jpg"),
        os.path.join(DATASET_FOLDER, split, frame_id.split("-")[0], frame_id.split("-")[-1] + "_CAM_FRONT.jpg"),
        os.path.join(DATASET_FOLDER, split, frame_id.split("-")[0], frame_id.split("-")[-1] + "_CAM_FRONT_RIGHT.jpg")
    ]

    px = list(data.past_states.pos_x)
    py = list(data.past_states.pos_y)

    xs = np.stack([px, py], axis=1)

    vx = list(data.past_states.vel_x)
    vy = list(data.past_states.vel_y)
    vs = np.stack([vx, vy], axis=1)

    ax = list(data.past_states.accel_x)
    ay = list(data.past_states.accel_y)
    acc = np.stack([ax, ay], axis=1)

    # from PIL import Image
    # image = Image.open(image_list[1])
    # image.save("1.jpg")

    return {
        "frame_id": frame_id,
        "image": image_list,
        "timestamp_micros": ts,
        "future waypoints": waypoints,
        "history waypoints": xs.tolist(),
        "velocity": vs.tolist(),
        "acceleration": acc.tolist(),
        "navigation_command": command[data.intent],
    }



def load_waymo_e2e_data():
    DATASET_FOLDER = '/weka/home/xliu316/scratchcxiao13/yingzi/workspace/waymo'
    TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training*.tfrecord*')
    VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val*.tfrecord*')
    TEST_FILES = os.path.join(DATASET_FOLDER, 'test*.tfrecord*')

    filenames = tf.io.matching_files(VALIDATION_FILES)
    if tf.size(filenames) == 0:
        raise FileNotFoundError(f"No TFRecords matched {VALIDATION_FILES}")

    cnt = 0
    # 顺序遍历所有 test tfrecord（可按需并行多个文件）

    gt_map = {}
    for f in filenames.numpy().tolist():
        ds = tf.data.TFRecordDataset(f, compression_type='')
        it = ds.as_numpy_iterator()
        for raw in tqdm(it, desc=f"Reading {os.path.basename(f)}"):
            data = wod_e2ed_pb2.E2EDFrame()
            data.ParseFromString(raw)

            if len(data.preference_trajectories) == 0 or \
                    data.preference_trajectories[0].preference_score == -1:
                continue

            gt_map[data.frame.context.name] = data
    
    return gt_map

import re
import re
from typing import List

# 支持 ±号 / 小数 / 科学计数法
NUM = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

# 抓取 "trajectory": "...." 的内容（允许换行、转义）
TRAJ_FIELD_RE = re.compile(
    r'"trajectory"\s*:\s*"(?P<inner>(?:\\.|[^"\\])*)"',
    flags=re.DOTALL
)

# 允许方括号被转义（\[, \]），逗号前也可能有反斜杠
PAIR_RE = re.compile(
    rf'\\?\[\s*({NUM})\s*\\?,\s*({NUM})\s*\\?\]',
    flags=re.IGNORECASE
)

def extract_trajectory(blob: str) -> List[List[float]]:
    """
    从任意长字符串中用正则提取 trajectory -> list[list[float,float]]。
    兼容：1) JSON中转义字符串；2) 直接出现的 [x,y]；3) 带 mdm 标记。
    """
    print(blob)
    blob = blob[blob.index("trajectory"):]

    # if m:
    #     inner = m.group('inner')
    # else:
    #     # 若没有标准 "trajectory": "..." 字段，就尝试 mdm 区块或全串
    #     mdm = re.search(r'<\|mdm_start\|>(.*?)<\|mdm_end\|>', blob, flags=re.DOTALL)
    #     inner = mdm.group(1) if mdm else blob

    # 去掉 mdm 标记
    inner = re.sub(r'<\|mdm_start\|>|<\|mdm_end\|>', '', blob)

    # 将常见的 JSON 转义去掉（只对可能影响解析的字符做“解转义”）
    # 例如 \" -> "， \[ -> [， \] -> ]， \, -> ,
    inner = (inner
             .replace(r'\"', '"')
             .replace(r'\[', '[')
             .replace(r'\]', ']')
             .replace(r'\,', ',')
             .replace(r'\+', '+'))

    # 提取 [x, y] 对
    pairs = PAIR_RE.findall(inner)

    if not pairs:
        # 再尝试一次：如果用户给的是未转义的原始文本，去掉上面的反斜杠替换可能带来的影响
        PAIR_RE_PLAIN = re.compile(rf'\[\s*({NUM})\s*,\s*({NUM})\s*\]', flags=re.IGNORECASE)
        pairs = PAIR_RE_PLAIN.findall(inner)

    if not pairs:
        raise ValueError("未能在 trajectory 中找到任何 [x, y] 坐标对。")

    return [[float(x), float(y)] for x, y in pairs]


def _finite_diff_velocity(p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    n = len(p); v = np.zeros(n, float)
    if n >= 2:
        v[0] = (p[1]-p[0])/(t[1]-t[0])
        v[-1] = (p[-1]-p[-2])/(t[-1]-t[-2])
    if n >= 3:
        v[1:-1] = (p[2:]-p[:-2])/(t[2:]-t[:-2])
    return v

def _finite_diff_accel(p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    n = len(p); a = np.zeros(n, float)
    if n < 3: return a
    # 端点用邻域近似
    a[0]  = 2*(((p[1]-p[0])/(t[1]-t[0])) - ((p[2]-p[1])/(t[2]-t[1]))) / ((t[1]-t[0]) + (t[2]-t[1]))
    a[-1] = 2*(((p[-1]-p[-2])/(t[-1]-t[-2])) - ((p[-2]-p[-3])/(t[-2]-t[-3]))) / ((t[-1]-t[-2]) + (t[-2]-t[-3]))
    for i in range(1, n-1):
        dt1 = t[i]-t[i-1]; dt2 = t[i+1]-t[i]
        a[i] = 2*(((p[i+1]-p[i])/dt2) - ((p[i]-p[i-1])/dt1)) / (dt1+dt2)
    return a

def _jmt_coeffs(p0, v0, a0, p1, v1, a1, T):
    A0 = p0
    A1 = v0
    A2 = a0/2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [  T3,    T4,     T5],
        [3*T2,  4*T3,   5*T4],
        [6*T,  12*T2,  20*T3]
    ], float)
    b = np.array([
        p1 - (A0 + A1*T + A2*T2),
        v1 - (A1 + 2*A2*T),
        a1 - (2*A2)
    ], float)
    A3, A4, A5 = np.linalg.solve(M, b)
    return np.array([A0, A1, A2, A3, A4, A5], float)

def _eval_quintic(coeffs, tau):
    a0,a1,a2,a3,a4,a5 = coeffs
    return (((a5*tau + a4)*tau + a3)*tau + a2)*tau**2 + a1*tau + a0

def jmt_interpolate_xy_with_start(p_start, traj_1to5, t_new):
    """
    p_start: 起点 (x0, y0)，对应 t=0s
    traj_1to5: 形如 (5,2) 的数组，依次是 t=1..5s 的 (x,y)
    t_new: 需要采样的时间戳（如 np.linspace(0.25, 5.0, 20)）
    """
    P = np.vstack([np.asarray(p_start, float)[None, :], np.asarray(traj_1to5, float)])  # (6,2)
    t = np.arange(0.0, 6.0)  # [0,1,2,3,4,5]  对应 P 的 6 个点

    # 估计端点导数
    vx = _finite_diff_velocity(P[:,0], t); vy = _finite_diff_velocity(P[:,1], t)
    ax = _finite_diff_accel  (P[:,0], t); ay = _finite_diff_accel  (P[:,1], t)

    # 为每段 [t_i, t_{i+1}] 生成 quintic（共 5 段：0-1,1-2,...,4-5）
    coeffs_x, coeffs_y, seg_starts = [], [], []
    for i in range(len(t)-1):
        T = t[i+1] - t[i]   # 这里就是 1s
        cx = _jmt_coeffs(P[i,0], vx[i], ax[i], P[i+1,0], vx[i+1], ax[i+1], T)
        cy = _jmt_coeffs(P[i,1], vy[i], ay[i], P[i+1,1], vy[i+1], ay[i+1], T)
        coeffs_x.append(cx); coeffs_y.append(cy); seg_starts.append(t[i])
    seg_starts = np.asarray(seg_starts)

    # 采样
    t_new = np.asarray(t_new, float)
    # 建议限制在 [0,5] 内，避免外推
    t_new = np.clip(t_new, 0.0, 5.0)

    X = np.empty_like(t_new); Y = np.empty_like(t_new)
    for k, tk in enumerate(t_new):
        i = min(np.searchsorted(seg_starts, tk, side='right')-1, len(seg_starts)-1)
        i = max(i, 0)
        tau = tk - seg_starts[i]
        X[k] = _eval_quintic(coeffs_x[i], tau)
        Y[k] = _eval_quintic(coeffs_y[i], tau)
    return np.stack([X, Y], axis=1)


def load_predictions(json_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    加载预测 JSON，返回：
      pred_map[frame_name] = {"trajs": [K,T,2], "probs": [K]}
    顶层可能叫 "prediction" 或 "predictions"，也可能直接是 {frame_name: ...}
    """
    with open(json_path, "r") as f:
        obj = json.load(f)

    from tqdm import tqdm
    import numpy as np
    pred_map = {}
    for line in tqdm(obj):
        try:
            trajectory = extract_trajectory(line['conversations'][-1]['value'])
            trajectory = trajectory[:5]

            scene = line['sample_id'].split("-")[0]
            from scipy.interpolate import PchipInterpolator

            # 已有的5个waypoints：1s,2s,3s,4s,5s
            traj = np.array(trajectory, dtype=float)  # shape (5,2)

            traj_4hz = jmt_interpolate_xy_with_start((0.0, 0.0), traj, np.linspace(0.25, 5.0, 20))
            print(traj_4hz.shape)
            pred_map[line['sample_id']] = traj_4hz
        except:
            pred_map[line['sample_id']] = np.zeros((20,2), dtype=float)


    return pred_map



def average_distance_per_step(predictions,
                              observed_traj,
                              mask,
                              time) -> np.ndarray:
    """
    ADE per proposal.
    predictions: [P, T, 2]
    observed_traj: [T, 2]
    mask: [T] bool
    return: [P]
    """
    if predictions.ndim == 2:
        predictions = predictions[None]  # -> [1, T, 2]

    observed_traj = observed_traj[None]
    mask = mask[None]

    predictions = predictions[:, :time, :]
    observed_traj = observed_traj[:, :time, :]
    mask = mask[:, :time]

    dist_per_step = np.linalg.norm(predictions - observed_traj, axis=-1)  # [P,T]
    dist_per_traj = (dist_per_step * mask[None]).sum(axis=-1)                  # [P]
    valid_steps = max(int(mask.sum()), 1)
    return dist_per_traj / float(valid_steps)


def main():
    ap = argparse.ArgumentParser(description="Waymo E2E Evaluation (ADE + RFS)")
    ap.add_argument("--pred_json", type=str, required=True,
                    help="Prediction JSON path (keys=frame_name).")
    args = ap.parse_args()


    prediction_dict = load_predictions(args.pred_json)
    gt_dict = load_waymo_e2e_data()
    gt_traj_dict = {}

    ade_3s_list = []
    ade_5s_list = []
    for frame_name in gt_dict:
        if frame_name not in prediction_dict:
            raise ValueError(f'No prediction for {frame_name}')
        data = gt_dict[frame_name]
        gt_traj = np.stack([data.future_states.pos_x, data.future_states.pos_y], axis=1)
        pred_traj = prediction_dict[frame_name]
        mask = np.ones(gt_traj.shape[0], dtype=np.bool_)
        gt_traj_dict[frame_name] = gt_traj


        ade_3s = average_distance_per_step(pred_traj[None], gt_traj, mask, 12)[0]
        ade_5s = average_distance_per_step(pred_traj[None], gt_traj, mask, 20)[0]
        ade_3s_list.append(ade_3s)
        ade_5s_list.append(ade_5s)


    rater_specified_trajectories = []
    rater_scores = []
    initial_speed = []

    prediction_trajectories = []
    prediction_probabilities = []
    frame_name_list = []
    gt_trajectories = []
    for frame_name, data in gt_dict.items():
        rater_specified_trajs_and_scores_i = data.preference_trajectories
        current_rater_trajs = []
        current_rater_scores = []
        for j in range(len(rater_specified_trajs_and_scores_i)):
            current_rater_trajs.append(
                np.stack(
                    [
                        rater_specified_trajs_and_scores_i[j].pos_x,
                        rater_specified_trajs_and_scores_i[j].pos_y,
                    ],
                    axis=-1,
                )
            )
            current_rater_scores.append(rater_specified_trajs_and_scores_i[j].preference_score)
        current_rater_scores = np.array(current_rater_scores)

        # Initial speed calculation. The last position of past velocity should be
        # the current velocity.
        vel_x = data.past_states.vel_x[-1]
        vel_y = data.past_states.vel_y[-1]
        initial_speed.append(np.sqrt(vel_x ** 2 + vel_y ** 2))

        # Fake prediction.
        prediction_traj = prediction_dict[frame_name]
        # We add an empty axis as the # of proposals is 1
        prediction_trajectories.append(prediction_traj[None])
        prediction_probabilities.append(np.ones(1))
        # Append the current trajectory and score to the batch list.
        rater_specified_trajectories.append(current_rater_trajs)
        rater_scores.append(current_rater_scores)
        frame_name_list.append(frame_name)
        gt_trajectories.append(gt_traj_dict[frame_name])

    # Convert the list of numpy array to add batch dimension.
    initial_speed = np.stack(initial_speed)
    prediction_trajectories = np.stack(prediction_trajectories)
    prediction_probabilities = np.stack(prediction_probabilities)

    rater_feedback_metrics = (
        get_rater_feedback_score(
            prediction_trajectories,
            prediction_probabilities,
            rater_specified_trajectories,
            rater_scores,
            initial_speed,
            frequency=4,  # Default is 4.
            length_seconds=5,  # Default predict 5 seconds.
            output_trust_region_visualization=False,
        )
    )
    rfs_score = rater_feedback_metrics['rater_feedback_score']

    rfs_score = rfs_score.tolist()


    def to_list_2dec(arr):
        """将任意形状的 numpy 数组四舍五入到两位小数并转为 list。"""
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = np.round(arr.astype(float), 2)
        return arr.tolist()

    pred_list = to_list_2dec(prediction_trajectories)
    gt_list = to_list_2dec(gt_trajectories)

    for pred, gt, name, score in zip(pred_list[-20:], gt_list[-20:], frame_name_list[-20:], rfs_score[-20:]):
        print(name)
        print("  score:", score)
        print("  pred:", pred)
        print("  gt  :", gt)

    print(f'ADE 3s: {np.mean(ade_3s_list)}')
    print(f'ADE 5s: {np.mean(ade_5s_list)}')
    print(f"RFS: {np.mean(rfs_score)}")


if __name__ == "__main__":
    main()


"""
python -m waymo.eval_planning \
    --pred_json ./waymo/ad_finetune_val_cot_planning_30k_ckpt1200.json

python -m waymo.eval_planning \
    --pred_json ./waymo/ad_finetune_val_cot_planning_30k_ckpt1000.json

python -m waymo.eval_planning \
    --pred_json /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/nuScenes/ad_finetune_waymo_val_cot_planning_ckpt1200_attack_v1.json

python -m waymo.eval_planning \
    --pred_json /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/nuScenes/ad_finetune_waymo_val_cot_planning_ckpt1000.json

python -m waymo.eval_planning \
    --pred_json /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/nuScenes/ad_finetune_waymo_val_cot_planning_ckpt1200_attack_v1.json

"""