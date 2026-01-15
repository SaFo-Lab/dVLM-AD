from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

from PIL import Image
import requests
import copy
import torch
import time
import json
import os

import sys
import warnings

import argparse

template = "\"{\\\"critical_objects\\\": {\\\"nearby_vehicle\\\": \\\"<|mdm_mask|>\\\", \\\"pedestrian\\\": \\\"<|mdm_mask|>\\\", \\\"cyclist\\\": \\\"<|mdm_mask|>\\\", \\\"construction\\\": \\\"<|mdm_mask|>\\\", \\\"traffic_element\\\": \\\"<|mdm_mask|>\\\", \\\"weather_condition\\\": \\\"<|mdm_mask|>\\\", \\\"road_hazard\\\": \\\"<|mdm_mask|>\\\", \\\"emergency_vehicle\\\": \\\"<|mdm_mask|>\\\", \\\"animal\\\": \\\"<|mdm_mask|>\\\", \\\"special_vehicle\\\": \\\"<|mdm_mask|>\\\", \\\"conflicting_vehicle\\\": \\\"<|mdm_mask|>\\\", \\\"door_opening_vehicle\\\": \\\"<|mdm_mask|>\\\"}, \\\"explanation\\\": \\\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\\\", \\\"future_meta_behavior\\\": {\\\"longitudinal\\\": \\\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\\\", \\\"lateral\\\": \\\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\\\"}, \\\"trajectory\\\": \\\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\\\"}\""

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    pretrained = "/weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/train/exp/llada-ad_finetune_waymo_cot_planning_30k"
    model_name = "llava_llada"
    device = "cuda:0"
    device_map = "cuda:0"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          attn_implementation="sdpa",
                                                                          device_map=device_map)


    image_paths = ["./CAM_FRONT_LEFT.jpg", "./CAM_FRONT.jpg", "./CAM_FRONT_RIGHT.jpg"]
    prompt = "You are an expert autonomous driving agent.\nTask 1: Critical Object Detection\nFor each class below, answer \"yes\" or \"no\" to indicate whether it affects the ego vehicle’s behavior or future trajectory:\n[nearby_vehicle, pedestrian, cyclist, construction, traffic_element, weather_condition, road_hazard, emergency_vehicle, animal, special_vehicle, conflicting_vehicle, door_opening_vehicle]\nTask 2: Scene Reasoning\nPredict the future behavior of the identified critical objects and explain how the identified critical objects or conditions affect the ego vehicle’s next 3-second trajectory.\nTask 3: Meta-Behavior Prediction\nPredict the ego vehicle’s future meta-driving behavior:\n- speed ∈ {keep, accelerate, decelerate, stop, other}\n- command ∈ {straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse, overtake, other}\nTask 4: Trajectory Prediction\nPredict the optimal 5-second future trajectory (5 waypoints, 1 s intervals).\n\nInput:\n- <image>: three front-view frames from left front, center front, right front cameras. \n- High-level navigation command: GO_STRAIGHT\n- Historical ego state: Provided are the previous ego vehicle status recorded over the last 3.0 seconds (at 0.5-second intervals).(t-3.0s) [-43.35, 0.12], Acceleration: X 0.04, Y 0.03 m/s², Velocity: X 14.28, Y -0.19 m/s,; (t-2.5s) [-36.21, 0.04], Acceleration: X 0.04, Y 0.04 m/s², Velocity: X 14.36, Y -0.12 m/s,; (t-2.0s) [-29.02, -0.02], Acceleration: X 0.06, Y 0.04 m/s², Velocity: X 14.43, Y -0.04 m/s,; (t-1.5s) [-21.81, -0.02], Acceleration: X 0.04, Y 0.04 m/s², Velocity: X 14.49, Y 0.04 m/s,; (t-1.0s) [-14.57, -0.01], Acceleration: X 0.04, Y 0.00 m/s², Velocity: X 14.55, Y 0.03 m/s,; (t-0.5s) [-7.30, 0.00], Acceleration: X 0.04, Y -0.02 m/s², Velocity: X 14.61, Y 0.00 m/s,; (t+0.0s) [0.00, 0.00], Acceleration: X 0.03, Y -0.01 m/s², Velocity: X 14.64, Y -0.01 m/s,\n\nOutput:"


    image_list = [Image.open(img) for img in image_paths]
    image_tensor = process_images(image_list, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llada"
    question = DEFAULT_IMAGE_TOKEN + f"""\n{prompt}"""

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    model.eval()
    if use_fast_dllm:
        register_fast_dllm_hook(model)
        print("Testing with Fast dLLM hook enabled")
    elif use_dllm_cache:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("Testing with cache enabled")
    else:
        print("Testing without cache")


    dynamic_decoding_kwargs = {
        "mdm_start_id": tokenizer.convert_tokens_to_ids("<|mdm_start|>"),
        "mdm_end_id": tokenizer.convert_tokens_to_ids("<|mdm_end|>"),
        "pad_id": tokenizer.pad_token_id,
        "min_field_len": 3,
        "end_prob_threshold": 0.60,
        "remain_ratio_gate": 1.0,
        "calibrate_end_logits": 0.0,
    }
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in image_list]


    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=128, gen_length=512, block_length=512, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
        prefix_refresh_interval=32,
        threshold=1,
        resp_template=template,
        **{
            "dynamic_decoding_config": dynamic_decoding_kwargs
        }
    )
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation time: {generation_time:.4f} seconds")

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)

    print(prompt_question)
    print(text_outputs)

