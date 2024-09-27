import argparse
import torch
from tqdm import tqdm
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
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
    # Model
    disable_torch_init()
    model_version = args.model_path.split("/")[-1]
    answers_file = args.answers_file
    model_name = get_model_name_from_path(args.model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    # Please add the bin file of the download UNIAA-LLaVA model bellows
    llava_model_path = [
        args.absolute_model_path + '/pytorch_model-00001-of-00003.bin',
        args.absolute_model_path + '/pytorch_model-00002-of-00003.bin',
        args.absolute_model_path + '/pytorch_model-00003-of-00003.bin',
    ]
    state_dict = {}
    for weight_file in llava_model_path:
        weight_file_path = weight_file
        state_dict_part = torch.load(weight_file_path, map_location=torch.device('cpu'))
        new_state_dict_part = {}
        for k, v in state_dict_part.items():
            new_state_dict_part[k] = v
        state_dict.update(new_state_dict_part)

    encoder_dict = {k:v for k, v in state_dict.items() if k.startswith('model.vision_tower.vision_tower.vision_model')}
    for old_key in list(encoder_dict.keys()):
        new_key = old_key.replace('model.vision_tower.vision_tower.vision_model.', 'vision_model.')
        encoder_dict[new_key] = encoder_dict.pop(old_key)
    vision_tower = model.get_vision_tower()
    vision_tower.vision_tower.load_state_dict(encoder_dict, strict=True)


    with open(args.questions_file) as f:
        llvqa_data = json.load(f)

    for i, llddata in enumerate(tqdm(llvqa_data)):
        filename = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError("IAA-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        if len(llddata["candidates"]) == 4:
            for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
                message += f"{choice} {ans}\n"
        elif len(llddata["candidates"]) == 2:
            for choice, ans in zip(["A.", "B."], llddata["candidates"]):
                message += f"{choice} {ans}\n"
        elif len(llddata["candidates"]) == 3:
            for choice, ans in zip(["A.", "B.", "C."], llddata["candidates"]):
                message += f"{choice} {ans}\n"
        elif len(llddata["candidates"]) == 5:
            for choice, ans in zip(["A.", "B.", "C.", "D.", "E."], llddata["candidates"]):
                message += f"{choice} {ans}\n"
        qs = message

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

        image = load_image(filename)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                num_beams=1,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        llddata["response"] = outputs

        with open(answers_file, "a") as wf:
            json.dump(llddata, wf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llava-v1.5")
    parser.add_argument("--model-path", type=str, default="zhouzhaokun/UNIAA-LLaVA")
    parser.add_argument("--absolute_model-path", type=str, default="path-to-UNIAA-LLaVA")

    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--questions-file", type=str,
                        default="UNIAA_QA.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args(--answers-file, type=str, default="UNIAA_QA_answers.json")
    eval_model(args)




