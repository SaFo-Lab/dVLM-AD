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

import sys
import warnings

mask_repeat = "<|mdm_mask|>" * 100

resp_template = r'''
{
  "critical_objects": {
    "nearby_vehicle": <|mdm_mask|>,
    "pedestrian": <|mdm_mask|>,
    "cyclist": <|mdm_mask|>,
    "construction": <|mdm_mask|>,
    "traffic_element": <|mdm_mask|>,
    "weather_condition": <|mdm_mask|>,
    "road_hazard": <|mdm_mask|>,
    "emergency_vehicle": <|mdm_mask|>,
    "animal": <|mdm_mask|>,
    "special_vehicle": <|mdm_mask|>,
    "conflicting_vehicle": <|mdm_mask|>,
    "door_opening_vehicle": <|mdm_mask|>
  },
  "explanation": <mask_repeat>,
  "meta_behaviour": {
    "speed": <|mdm_mask|>,
    "command": <|mdm_mask|><|mdm_mask|><|mdm_mask|>
  },
  "trajectory": [
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
    [<|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>, <|mdm_mask|><|mdm_mask|>.<|mdm_mask|><|mdm_mask|>],
  ]
}<|eot_id|>'''

resp_template = resp_template.replace("<mask_repeat>", mask_repeat).replace("\n", "")

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
pretrained = "/weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/train/exp/llada-ad_finetune_planning"
# pretrained = "GSAI-ML/LLaDA-V"
pretrained = "/weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/train/exp/llada-ad_finetune"

model_name = "llava_llada"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args

print(tokenizer("<|mdm_end|>"))
print(tokenizer("<|mdm_start|>"))
# <|mdm_mask|>

import sys
sys.exit(0)
model.eval()
image = Image.open("/weka/home/xliu316/scratchcxiao13/yingzi/workspace/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg")
image.save("1.png")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 

question = r"""You are an expert autonomous driving agent.

Input:
- <image>: One front-view image captured from the ego-vehicle at the current timestep.
- Current high-level driving intent: Turn left.
- Previous ego-vehicle status recorded at 0.5-second intervals. Each record includes
  the x and y coordinates of the ego vehicle, acceleration (X, Y) in m/s², velocity (m/s),
  and steering angle (radians; positive for left turns, negative for right turns).
  The data is presented in the format:
  [x, y]: (t-0.0s) [0.0, 0.0], Acceleration: X 0.44, Y 0.44 m/s², Velocity: 9.19 m/s, Steering angle: 0.36 (positive: left turn, negative: right turn)

Task 1: Critical Objects and Conditions Detection
Determine which of the following object classes may influence the ego vehicle’s behavior
or future trajectory. Answer strictly with "yes" or "no" for each class.
Object classes:
- nearby_vehicle
- pedestrian
- cyclist
- construction
- traffic_element
- weather_condition
- road_hazard
- emergency_vehicle
- animal
- special_vehicle
- conflicting_vehicle
- door_opening_vehicle

Task 2: Scene Reasoning and Explanation
Provide a concise explanation (~100 words) describing why the identified critical
objects or conditions influence the ego vehicle’s next 3-second trajectory.
Focus on the interaction between the ego vehicle and its surroundings,
including motion, traffic rules, and potential conflicts.

Task 3: Meta-Behaviour Prediction
Classify the intended meta-driving behaviour of the ego vehicle based on scene context:
- speed: {keep, accelerate, decelerate, other}
- command: {straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse, other}

Task 4: Trajectory Prediction
Predict the optimal 3-second future trajectory of the ego vehicle (6 waypoints at 0.5s intervals).
Each waypoint should be a pair of coordinates [x, y] in meters relative to the ego vehicle,
where +x indicates forward motion and +y indicates leftward motion.

Output Format:
Return a single JSON object (strict JSON, no extra keys, no commentary):"""

# question = "You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last 0.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:.(t-0.0s) [0.0, 0.0], Acceleration: X 0.44, Y 0.44 m/s^2, Velocity: 9.19 m/s, Steering angle: 0.36 (positive: left turn, negative: right turn)\n"
# resp_template = None

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

print(prompt_question)

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



print(resp_template)

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

start_time = time.time()
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=128, gen_length=384, block_length=384, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
    prefix_refresh_interval=64,
    threshold=1,
    resp_template=resp_template,
)
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

print(cont.shape)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)
