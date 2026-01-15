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


prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")



def main(model, config):

    with open(config.data_file, "rb") as f:
        data = json.load(f)

    import random
    from tqdm import tqdm
    random.seed(2333)
    random.shuffle(data)
    outputs = []
    for i, line in tqdm(enumerate(data), total=len(data)):
        if i < 510:
            continue
        if i >= 1000:
            break
        image_list = [Image.open(img) for img in line["image"]]
        image_tensor = process_images(image_list, image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "llava_llada"
        question = DEFAULT_IMAGE_TOKEN + f"""\n{line['conversations'][0]['value']}"""

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

        # line['conversations'][-1]['value'] = line['conversations'][-1]['value'].replace("\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\"",
        #                                            "\"<|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|>\"")

        start_time = time.time()
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=128, gen_length=512, block_length=512, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
            prefix_refresh_interval=32,
            threshold=1,
            resp_template=line['conversations'][-1]['value'],
            **{
                "dynamic_decoding_config": dynamic_decoding_kwargs
            }
        )
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Generation time: {generation_time:.4f} seconds")

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
        line['conversations'][-1]['value'] = text_outputs[0]

        if not os.path.exists(config.result_file):
            outputs = []
        else:
            with open(config.result_file, "r") as f:
                outputs = json.load(f)

        outputs.append(line)

        # import sys
        # sys.exit(0)

        with open(config.result_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of planning')
    parser.add_argument('--model_path', type=str, help='name of the method being evaluated, used for table print', default='Agent-Driver')
    parser.add_argument('--data_file', type=str, help='path to the result file', default='temp_results/refined_trajs_dict_0.0_5.0_1.265_7.89.pkl')
    parser.add_argument('--result_file', type=str, help='path to the result file', default='temp_results/refined_trajs_dict_0.0_5.0_1.265_7.89.pkl')
    parser.add_argument('--metric', type=str, default='uniad', help='metric to evaluate, either uniad or stp3')
    parser.add_argument('--gt_dir', type=str, default='nuscenes/metrics')
    config = parser.parse_args()

    pretrained = config.model_path
    model_name = "llava_llada"
    device = "cuda:0"
    device_map = "cuda:0"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          attn_implementation="sdpa",
                                                                          device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.half()
    model.eval()

    main(model, config)


"""
python ./nuScenes/inference.py \
    --model_path /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/train/exp/llada-ad_finetune_cot_planning_28k \
    --data_file /weka/home/xliu316/scratchcxiao13/yingzi/workspace/dvlm/dvlm-ad_nusc_val_cot_v1.json \
    --result_file /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/eval/nuScenes/ad_finetune_cot_planning_28k_final_v1.json 
    
"""