import argparse
import torch
from tqdm import tqdm
import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from PIL import Image
from io import BytesIO


device = "cuda" if torch.cuda.is_available() else "cpu"
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image




disable_torch_init()
model_name = "llava-v1.5"
tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint, None, model_name)
question_file = "UNIAA_Describe.json"
output_file = "UNIAA_Describe_output.json"
output_dir = 'UNIAA-Bench/'
output_file = output_dir + '/' + output_file


llava_model_path = [
    checkpoint + '/pytorch_model-00001-of-00003.bin',
    checkpoint + '/pytorch_model-00002-of-00003.bin',
    checkpoint + '/pytorch_model-00003-of-00003.bin'
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



with open(question_file) as f:
    description_data = json.load(f)

for i, data in enumerate(tqdm(description_data)):
    filename = data["img_path"]
    qs = data["question"]
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

    image = load_image(filename)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            # num_beams=1,
            do_sample=True,
            temperature=0.1,
            top_p=0.7,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    data["response"] = outputs
    #将字典按序写入json中
    with open(output_file, "a") as wf:
        json.dump(data, wf)