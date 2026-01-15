import os
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .utils import get_rater_feedback_score
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e_data_pb2
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from PIL import Image
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import os, json, numpy as np, tensorflow as tf
from tqdm import tqdm


DATASET_FOLDER = '/weka/home/xliu316/scratchcxiao13/yingzi/workspace/waymo'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training*.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val*.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test*.tfrecord*')

import re
import re
from typing import List


NUM = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'


TRAJ_FIELD_RE = re.compile(
    r'"trajectory"\s*:\s*"(?P<inner>(?:\\.|[^"\\])*)"',
    flags=re.DOTALL
)


PAIR_RE = re.compile(
    rf'\\?\[\s*({NUM})\s*\\?,\s*({NUM})\s*\\?\]',
    flags=re.IGNORECASE
)

def extract_trajectory(blob: str) -> List[List[float]]:
    blob = blob[blob.index("trajectory"):]
    inner = re.sub(r'<\|mdm_start\|>|<\|mdm_end\|>', '', blob)


    inner = (inner
             .replace(r'\"', '"')
             .replace(r'\[', '[')
             .replace(r'\]', ']')
             .replace(r'\,', ',')
             .replace(r'\+', '+'))


    pairs = PAIR_RE.findall(inner)

    if not pairs:
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
    P = np.vstack([np.asarray(p_start, float)[None, :], np.asarray(traj_1to5, float)])  # (6,2)
    t = np.arange(0.0, 6.0)

    vx = _finite_diff_velocity(P[:,0], t); vy = _finite_diff_velocity(P[:,1], t)
    ax = _finite_diff_accel  (P[:,0], t); ay = _finite_diff_accel  (P[:,1], t)


    coeffs_x, coeffs_y, seg_starts = [], [], []
    for i in range(len(t)-1):
        T = t[i+1] - t[i]
        cx = _jmt_coeffs(P[i,0], vx[i], ax[i], P[i+1,0], vx[i+1], ax[i+1], T)
        cy = _jmt_coeffs(P[i,1], vy[i], ay[i], P[i+1,1], vy[i+1], ay[i+1], T)
        coeffs_x.append(cx); coeffs_y.append(cy); seg_starts.append(t[i])
    seg_starts = np.asarray(seg_starts)


    t_new = np.asarray(t_new, float)

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

    with open(json_path, "r") as f:
        obj = json.load(f)

    with tf.io.gfile.GFile("/weka/home/xliu316/scratchcxiao13/yingzi/RAP/MySubmission/mysubmission.binproto-00000-of-00001", 'rb') as fp:
        data = fp.read()
        shard = wod_e2ed_submission_pb2.E2EDChallengeSubmission()
        shard.ParseFromString(data)

    gt_dict = {}
    for line in shard.predictions:
        idx = [3, 7, 11, 15, 19]
        gt_dict[line.frame_name] = [[float(x), float(y)] for x, y in zip(line.trajectory.pos_x, line.trajectory.pos_y)]
        gt_dict[line.frame_name] = [gt_dict[line.frame_name][i] for i in idx]


    from tqdm import tqdm
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    pred_map = {}
    cnt = 0
    outputs = []
    for line in tqdm(obj):
        trajectory = extract_trajectory(line['conversations'][-1]['value'])
        trajectory = trajectory[:5]
        traj = np.array(trajectory, dtype=float)
        trajectory = extract_trajectory(line['conversations'][-1]['value'])
        trajectory = trajectory[:5]
        scene = line['sample_id'].split("-")[0]
        traj = np.array(trajectory, dtype=float)  # shape (5,2)
        traj_4hz = jmt_interpolate_xy_with_start((0.0, 0.0), traj, np.linspace(0.25, 5.0, 20))
        print(traj_4hz.shape)
        pred_map[line['sample_id']] = traj_4hz

    with open("./waymo/ad_finetune_test_cot_planning_30k_final_v2.json", "w") as f:
        json.dump(outputs, f)


    print(cnt)
    # import sys
    # sys.exit(0)


    return pred_map

def extract_waypoints_from_frame_bytes(frame_bytes):
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(frame_bytes)

    if len(data.preference_trajectories) == 0 or \
            data.preference_trajectories[0].preference_score == -1:
        return None

    return data

filenames = tf.io.matching_files(TEST_FILES)
if tf.size(filenames) == 0:
    raise FileNotFoundError(f"No TFRecords matched {TEST_FILES}")


pred_file = ""
pred_map = load_predictions(pred_file)


predictions = []
from tqdm import tqdm
import math
for frame_name, traj in tqdm(pred_map.items()):
        predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(pos_x=traj[:, 0],
                                                                            pos_y=traj[:, 1])


        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(frame_name=frame_name, trajectory=predicted_trajectory)
        # The final prediction should be a list of FrameTrajectoryPredictions.
        predictions.append(frame_trajectory)


print(len(predictions))
num_submission_shards = 1  # Please modify accordingly.
submission_file_base = '/weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/waymo/MySubmission'  # Please modify accordingly.
if not os.path.exists(submission_file_base):
  os.makedirs(submission_file_base)

sub_file_names = [
    os.path.join(submission_file_base, "mysubmission.binproto-00000-of-00001")
]


print(sub_file_names)
# As the submission file may be large, we shard them into different chunks.
submissions = []
num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
for i in range(num_submission_shards):
  start = i * num_predictions_per_shard
  end = (i + 1) * num_predictions_per_shard
  submissions.append(
      wod_e2ed_submission_pb2.E2EDChallengeSubmission(
          predictions=predictions[start:end]))

print(len(submissions))
# print(submissions[0])

for i, shard in enumerate(submissions):
  shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
  shard.authors[:] = ['YingziMa']  # Please modify accordingly.
  shard.affiliation = 'Wisc'  # Please modify accordingly.
  shard.account_name = 'yma382@wisc.edu'  # Please modify accordingly.
  shard.unique_method_name = 'dVLM-AD'  # Please modify accordingly.
  shard.method_link = 'https://vlm-driver.github.io/'  # Please modify accordingly.
  shard.description = 'dVLM for planning'  # Please modify accordingly.
  shard.uses_public_model_pretraining = True # Please modify accordingly.
  shard.public_model_names.extend(['Model_name']) # Please modify accordingly.
  shard.num_model_parameters = "7B" # Please modify accordingly.

  with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
    fp.write(shard.SerializeToString())

"""
python -m waymo.submission
cd /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/waymo
tar cvf MySubmission.tar MySubmission
gzip MySubmission.tar

"""