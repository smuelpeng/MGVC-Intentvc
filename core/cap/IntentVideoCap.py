import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import math
from datetime import timedelta
import numpy as np
import decord
from decord import VideoReader
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import os
import json
from multiprocessing import Pool
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
import datetime
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt
from video_aug import VideoAugmentation
from prompt_util import IntentPromptSchema
from tqdm import tqdm

def load_captions(caption_path):
    with open(caption_path, 'r') as f:
        captions = json.load(f)['captions']
    return captions


class IntentVideoCap:
    def __init__(self, stage1_model_path, stage2_model_path):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda"
        }
        self.model_stage1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(stage1_model_path, **model_kwargs)
        self.model_stage2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(stage2_model_path, **model_kwargs)

        self.processor = AutoProcessor.from_pretrained(stage1_model_path)
        self.video_aug = VideoAugmentation()
        self.prompt_schema = IntentPromptSchema()


        self.dam_captions = load_captions('data/base_captions/DAM.json')
        self.intervl_captions = load_captions('data/base_captions/InternVL.json')
        self.qwen_captions = load_captions('data/base_captions/Qwen.json')

    def resize_image_with_aspect_ratio(self, image, max_size=1024):
        """Resize image keeping aspect ratio so that the longest edge is max_size and dimensions are multiples of 28."""
        width, height = image.size
        
        # Calculate initial resize dimensions
        if width > height:
            new_width = min(max_size, width)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(max_size, height)
            new_width = int(width * (new_height / height))
        
        # Adjust dimensions to be multiples of 28
        new_width = (new_width // 28) * 28
        new_height = (new_height // 28) * 28
        
        # Ensure minimum size
        new_width = max(28, new_width)
        new_height = max(28, new_height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def get_base_caption(self, video_id):
        dam_caption = self.dam_captions.get(video_id, "")
        intervl_caption = self.intervl_captions.get(video_id, "")
        qwen_caption = self.qwen_captions.get(video_id, "")
        base_caption = f'Here is a related caption for this video: {dam_caption}\n\n'
        base_caption += f'{intervl_caption}\n\n'
        base_caption += f'{qwen_caption}\n\n'
        return base_caption

    def get_slot_prompt(self, stage1_caption):
        stage1_caption = stage1_caption if isinstance(stage1_caption, str) else stage1_caption[0]
        slot_prompt = f"Refine the following caption:\n"
        slot_prompt += f"{stage1_caption}\n\n"
        slot_prompt += "**Your caption must include the following semantic slots**:\n"
        slot_prompt += "- [subject]: the main object in the red box, with visible attributes\n"
        slot_prompt += "- [action]: its main motion or activity\n"
        slot_prompt += "- [location]: where the action takes place (e.g., grassy field, snowy ground, inside cage)\n"
        slot_prompt += "- [scene_context] (optional): surrounding scene or environment, only if relevant\n"
        slot_prompt += "- [interaction_target] (optional): what it interacts with, if applicable\n"
        slot_prompt += "- [manner] (optional): how the action is performed (e.g., steadily, playfully)\n\n"                    
        return slot_prompt
    
    @torch.no_grad()
    def cap(self, video_path, bbox_path, num_frames=36, max_new_tokens=512, video_size=448):
        '''
        get video caption from video and bbox, for three augmentations
        '''
        results = {}
        aug_types = {
            "base": self.video_aug.get_basic_video_aug,
            "maxboundbox": self.video_aug.get_crop_video_max_bound_bbox,
            "centerbox": lambda v, b: self.video_aug.get_cropped_video_center_bbox(v, b, scale=2.0)
        }
        video_id = os.path.basename(video_path).split(".")[0]
        category = os.path.basename(video_path).split("-")[0]

        caption_list = []
        for aug_name, aug_func in aug_types.items():
            frames, bboxes = aug_func(video_path, bbox_path)
            if not frames:
                results[aug_name] = ""
                continue
            # 均匀采样num_frames帧
            idxs = np.linspace(0, len(frames)-1, num=min(num_frames, len(frames)), dtype=int)
            sampled_frames = [frames[i] for i in idxs]
            # 预处理
            frame_list = []
            # video_out = cv2.VideoWriter(f"temp_{aug_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 1, (frames[0].shape[1], frames[0].shape[0]))
            for frame in sampled_frames:

                # save video
                # out_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # video_out.write(out_frame)

                pil_img = Image.fromarray(frame)
                pil_img = self.resize_image_with_aspect_ratio(pil_img, max_size=video_size)
                buffer = BytesIO()
                pil_img.save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                frame_list.append(f"data:image/jpeg;base64,{img_str}")
            # video_out.release()


            # stage1
            # 构造prompt
            system_content = self.prompt_schema.get_prompt(category)['system_content']
            user_content = self.prompt_schema.get_prompt(category)['core_prompt']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "video", "video": frame_list, "fps": 1.0},
                    {"type": "text", "text": user_content}
                ]}
            ]
            # 推理
            text_stage1 = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_stage1],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model_stage1.device)
            generated_ids = self.model_stage1.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            caption = caption.split("\n")
            results[aug_name] = [c.strip() for c in caption if c.strip()]
            caption_list += results[aug_name]

            # stage2
            user_content = user_content.replace("Generate one English caption", f"Generate five English captions")

            base_caption = self.get_base_caption(video_id)
            user_content = f'{user_content}\n\n{base_caption}'
            # stage1_caption =f'please generate five captions for this video: {results[aug_name]}'

            slot_prompt = self.get_slot_prompt(results[aug_name])
            user_content = f'{user_content}\n\n{slot_prompt}'


            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "video", "video": frame_list, "fps": 1.0},
                    {"type": "text", "text": user_content}
                ]}
            ]
            text_stage2 = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_stage2],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model_stage2.device)
            generated_ids = self.model_stage2.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            caption = caption.split("\n")
            results[aug_name] = [c.strip() for c in caption if c.strip()]
            caption_list += results[aug_name]

        return text_stage1, text_stage2, caption_list
    
if __name__ == "__main__":
    # video_id = "rabbit-16"
    INPUT_VIDEO_PATH = "data/intentvc/videos/"


    STAGE1_MODEL_PATH = "models/cap_stage1"
    STAGE2_MODEL_PATH = "models/cap_stage2"

    intent_video_cap = IntentVideoCap(STAGE1_MODEL_PATH, STAGE2_MODEL_PATH)

    with open(f"data/sample_result_public.json", "r") as f:
        data = json.load(f)['captions']

    cap_results = {}
    num_frames = 12
    max_new_tokens = 512
    video_size = 448
    for video_id, captions in tqdm(data.items(), total=len(data)):
        category = video_id.split("-")[0]
        video_path = os.path.join(INPUT_VIDEO_PATH, f"{category}/{video_id}/{video_id}.mp4")
        bbox_path = os.path.join(INPUT_VIDEO_PATH, f"{category}/{video_id}/object_bboxes.txt")
        text_stage1, text_stage2, results = intent_video_cap.cap(video_path, bbox_path, num_frames=num_frames, max_new_tokens=max_new_tokens, video_size=video_size)
        cap_results[video_id] = results

    with open(f"data/sample_result_public_captioned.json", "w") as f:
        json.dump({"version": "v1", "captions": cap_results}, f, indent=2)

