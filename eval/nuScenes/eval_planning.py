# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import pickle
import os
import numpy as np
from pyquaternion import Quaternion
import json
from os import path as osp
from .planning_utils import PlanningMetric
import torch
from tqdm import tqdm
import threading
import re


def append_tangent_directions(traj):
    directions = []
    directions.append(np.arctan2(traj[0][1], traj[0][0]))
    for i in range(1, len(traj)):
        vector = traj[i] - traj[i - 1]
        angle = np.arctan2(vector[1], vector[0])
        directions.append(angle)
    directions = np.array(directions).reshape(-1, 1)
    traj_yaw = np.concatenate([traj, directions], axis=-1)
    return traj_yaw


def print_progress(current, total):
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.2f}%)", end="")


def process_data(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric):
    ego_boxes = np.array([[0.5 + 0.985793, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])
    for i in range(start, end):

        data = key_infos['infos'][i]
        if data['token'] not in preds.keys():
            continue



        pred_traj = preds[data['token']]
        gt_traj, mask = data['gt_planning'], data['gt_planning_mask'][0]


        gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
        gt_agent_feats = np.concatenate(
            [data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6],
             data['gt_fut_idx']], -1)
        bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

        e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
        e2g_t = data['ego2global_translation']
        drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)
        pred_traj_yaw = append_tangent_directions(pred_traj[..., :2])
        pred_traj_mask = np.concatenate(
            [pred_traj_yaw[..., :2].reshape(1, -1), np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1),
             pred_traj_yaw[..., 2:].reshape(1, -1)], axis=-1)
        ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)

        pred_traj = torch.from_numpy(pred_traj).unsqueeze(0)
        gt_traj = torch.from_numpy(gt_traj[..., :2])

        # print("="*50)
        # print(pred_traj)
        # print(gt_traj)


        fut_valid_flag = mask.all()
        future_second = 3
        if fut_valid_flag:
            with lock:
                metric_dict['samples'] += 1
            for i in range(future_second):
                cur_time = (i + 1) * 2
                ade = float(
                    sum(
                        np.sqrt(
                            (pred_traj[0, j, 0] - gt_traj[0, j, 0]) ** 2
                            + (pred_traj[0, j, 1] - gt_traj[0, j, 1]) ** 2
                        )
                        for j in range(cur_time)
                    )
                    / cur_time
                )
                metric_dict['l2_{}s'.format(i + 1)] += ade

                obj_coll, obj_box_coll = planning_metric.evaluate_coll(pred_traj[:, :cur_time],
                                                                       gt_traj[:, :cur_time],
                                                                       torch.from_numpy(bev_seg[1:]).unsqueeze(0))


                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] += obj_box_coll.max().item()

                rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0
                out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (
                            ego_seg[1:cur_time + 1] == 1)).sum() > 0
                if out_of_drivable and not rec_out:
                    metric_dict['plan_boundary_{}s'.format(i + 1)] += 1

        pbar.update(1)



def main(args):
    pred_path = args.pred_path
    anno_path = args.anno_path
    key_infos = pickle.load(open(osp.join(args.base_path, anno_path), 'rb'))
    preds = dict()
    # for data in key_infos['infos']:
    #     if os.path.exists(pred_path + data['token']):
    #         with open(pred_path + data['token'], 'r', encoding='utf8') as f:
    #             pred_data = json.load(f)
    #             traj = pred_data[0]['A'][0]
    #             full_match = re.search(r'\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', traj)
    #             if full_match:
    #                 coordinates_matches = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full_match.group(0))
    #                 coordinates = [tuple(map(float, re.findall(r'-?\d+\.\d+', coord))) for coord in coordinates_matches]
    #                 coordinates_array = np.array(coordinates)
    #                 preds[data['token']] = coordinates_array

    def load_pred_trajs_from_file(path, token_list):
        with open(path, "rb") as f:
            pred_data = json.load(f)

        # pred_data = pred_data[-62:]

        # NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?'
        # PAIR_RE = re.compile(r'\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]', re.IGNORECASE)
        #
        # def process(text):
        #     matches = PAIR_RE.findall(text)
        #     matches = matches[:6]  # 只取前 6 个点
        #     arr = np.array([[float(x), float(y)] for x, y in matches], dtype=float)
        #     return arr

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

        def process(blob: str) -> List[List[float]]:
            """
            从任意长字符串中用正则提取 trajectory -> list[list[float,float]]。
            兼容：1) JSON中转义字符串；2) 直接出现的 [x,y]；3) 带 mdm 标记。
            """
            # print(blob)
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

            pairs = pairs[:6]
            return np.array([[float(x), float(y)] for x, y in pairs], dtype=float)

        # with open("./nuScenes/llada-ad_nusc_v2.json", "rb") as f:
        #     demo_data = json.load(f)
        #
        # demo_trajs_dict = {line['sample_id'].split("_")[-1]: process(line['conversations'][-1]['value']) for line in
        #                    demo_data if line['sample_id'].split("_")[-1] in token_list}
        #
        # demo_trajs_dict = {k: v for k, v in demo_trajs_dict.items() if v.shape == (6, 2)}
        #

        pred_trajs_dict = {line['sample_id'].split("_")[-1]: process(line['conversations'][-1]['value']) for line in
                           pred_data if line['sample_id'].split("_")[-1] in token_list}

        pred_trajs_dict = {k: v for k, v in pred_trajs_dict.items() if v.shape == (6, 2)}

        print(len(pred_trajs_dict.keys()))
        return pred_trajs_dict

    preds = load_pred_trajs_from_file(pred_path, [item['token'] for item in key_infos['infos']])



    metric_dict = {
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
        'plan_boundary_1s': 0,
        'plan_boundary_2s': 0,
        'plan_boundary_3s': 0,
        'l2_1s': 0,
        'l2_2s': 0,
        'l2_3s': 0,
        'samples': 0,
    }

    num_threads = args.num_threads
    total_data = len(key_infos['infos'])
    data_per_thread = total_data // num_threads
    threads = []
    lock = threading.Lock()
    pbar = tqdm(total=total_data)
    for i in range(num_threads):
        start = i * data_per_thread
        end = start + data_per_thread
        if i == num_threads - 1:
            end = total_data
        thread = threading.Thread(target=process_data,
                                  args=(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    pbar.close()
    for k in metric_dict:
        if k != "samples":
            print(f"""{k}: {metric_dict[k] / metric_dict["samples"]}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('--base_path', type=str, default='../data/nuscenes/', help='Base path to the data.')
    parser.add_argument('--pred_path', type=str, default='results_planning_only/',
                        help='Path to the prediction results.')
    parser.add_argument('--anno_path', type=str, default='nuscenes2d_ego_temporal_infos_val.pkl',
                        help='Path to the annotation file.')
    parser.add_argument('--num_threads', type=int, default=32, help='Number of threads to use.')

    args = parser.parse_args()

    planning_metric = PlanningMetric(args.base_path)
    main(args)

"""
python -m nuScenes.eval_planning \
    --base_path /weka/home/xliu316/scratchcxiao13/yingzi/workspace/nuscenes \
    --pred_path ./nuScenes/ad_finetune_cot_planning_28k_final_v1.json \
    --anno_path nuscenes2d_ego_temporal_infos_val.pkl 
    
python -m nuScenes.eval_planning \
    --base_path /weka/home/xliu316/scratchcxiao13/yingzi/workspace/nuscenes \
    --pred_path ./nuScenes/ad_finetune_nusc_cot_planning_28k_ckpt1200.json \
    --anno_path nuscenes2d_ego_temporal_infos_val.pkl 
    
python -m nuScenes.eval_planning \
    --base_path /weka/home/xliu316/scratchcxiao13/yingzi/workspace/nuscenes \
    --pred_path ./nuScenes/llada-ad_nusc_v2.json \
    --anno_path nuscenes2d_ego_temporal_infos_val.pkl 
"""