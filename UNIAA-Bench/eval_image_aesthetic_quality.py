import argparse
import torch

import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from tqdm import tqdm

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt += " The aesthetic quality is"

    import json

    with open("llava_v1_5_ava_scores.json") as f:
        data = json.load(f)

    out_lines = []
    for i, llddata in enumerate(tqdm(data)):
        image_path = llddata["image"]
        iaa_score = llddata["iaa_score"]

        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_logits = model(input_ids,
                images=image_tensor)["logits"][:,-1]

        probs, inds = output_logits.sort(dim=-1, descending=True)
        #print(probs[0, 0:100], inds[0, 0:100], tokenizer.convert_ids_to_tokens(inds[0, 0:100]))
        # 1781: good, 6460: poor
        # 1880: high, 4482: low
        # 15129: excellent, 1781: good, 6534: fair, 6460: poor, 4319: bad
        '''
        lgood, lpoor = output_logits[0,1781].item(), output_logits[0,6460].item()
        lhigh, llow = output_logits[0,1880].item(), output_logits[0,4482].item()
        llddata["logit_good"] = lgood
        llddata["logit_poor"] = lpoor
        llddata["logit_high"] = lhigh
        llddata["logit_low"] = llow
        out_lines.append(" ".join([image_path, str(iaa_score),
                                   str(float(llddata["logit_good"])), str(float(llddata["logit_poor"])),
                                   str(float(llddata["logit_high"])), str(float(llddata["logit_low"])),
                                   ]) + "\n")
        '''
        lexcel, lgood, lfair, lpoor, lbad = output_logits[0,15129].item(), output_logits[0,1781].item(), output_logits[0,6534].item(), output_logits[0,6460].item(), output_logits[0,4319].item()
        out_lines.append(" ".join([image_path, str(iaa_score),
                                   str(float(lexcel)), str(float(lgood)), str(float(lfair)), str(float(lpoor)), str(float(lbad))])
                                   + "\n")
    f = open(os.path.join(args.model_path, "zero-shot-iaa-score-ava-5gear.txt"), "w")
    f.writelines(out_lines)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/interns/zhouzhaokun/LLaVA/checkpoints/llava-v1.5-7b-kaa-lora-v2-v100-pccd_ava_46k-5class-bad-1epoch")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--query", type=str, default="Rate this image from an aesthetic perspective.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    args = parser.parse_args()
    eval_model(args)
