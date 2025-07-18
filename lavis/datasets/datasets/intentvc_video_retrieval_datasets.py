"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import os
import json

from collections import OrderedDict

from PIL import Image
from mmengine.fileio import FileClient 

from lavis.datasets.datasets.base_dataset import BaseDataset




class IntentVCVideoRetrievalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """

        self.annotation = []
        self.vis_root = vis_root


        self.img_ids = {}
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root

        annotation_dict = json.load(open(ann_paths[0]))['captions']
        anno_num = 0
        video_n = 0
        for video_id, v in annotation_dict.items():
            category = video_id.split('-')[0]

            if video_id not in self.img_ids:
                self.img_ids[video_id] = video_n
                video_n += 1

            for i, caption in enumerate(v):
                self.annotation.append({
                    "image_id": video_id,
                    "caption": caption,
                    "id": anno_num,
                    "video": f'{vis_root}/{category}/{video_id}/{video_id}.mp4'
                })
                anno_num += 1
        if os.path.exists(f'{vis_root}/lock.mdb'):
            self.file_client = FileClient(backend='lmdb', db_path=vis_root)
        else:
            self.file_client = None

    def __getitem__(self, index):
        ann = self.annotation[index]
        if self.file_client is not None:
            vid_obj = io.BytesIO(self.file_client.get(ann["video"]))
        else:
            if not os.path.exists(ann["video"]):
                video_id = ann["image_id"]
                video_path = f'{self.vis_root}/{video_id}.mp4'
                vid_obj = io.BytesIO(open(video_path, 'rb').read())
            else:
                vid_obj = io.BytesIO(open(ann["video"], 'rb').read())

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])

        return {
            "video": video,
            "text_input": caption,
            "video_id": self.img_ids[ann["image_id"]],
        }


class IntentVCVideoRetrievalEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Forge annotations
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.vis_root = vis_root
        # self.ann_paths = ann_paths
        self.annotation = []
        self.img_ids = {}

        annotation_dict = json.load(open(ann_paths[0]))['captions']
        anno_num = 0
        video_n = 0
        for video_id, v in annotation_dict.items():     
            category = video_id.split('-')[0]

            if video_id not in self.img_ids:
                self.img_ids[video_id] = video_n
                video_n += 1

            for i, caption in enumerate(v[:1]):
                self.annotation.append({
                    "image_id": video_id,
                    "video": f'{vis_root}/{category}/{video_id}/{video_id}.mp4',
                    "caption": caption,
                    "instance_id": anno_num,
                    })
                anno_num += 1

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.text.append(self.text_processor(ann["caption"]))
            self.img2txt[img_id] = [txt_id]
            self.txt2img[txt_id] = img_id
            txt_id += 1


        if os.path.exists(f'{vis_root}/lock.mdb'):
            self.file_client = FileClient(backend='lmdb', db_path=vis_root)
        else:
            self.file_client = None            

    def __getitem__(self, index):
        ann = self.annotation[index]
        if self.file_client is not None:
            vid_obj = io.BytesIO(self.file_client.get(ann["video"]))
        else:   
            if not os.path.exists(ann["video"]):
                video_id = ann["image_id"]
                video_path = f'{self.vis_root}/{video_id}.mp4'
                vid_obj = io.BytesIO(open(video_path, 'rb').read())
            else:
                vid_obj = io.BytesIO(open(ann["video"], 'rb').read())

        video = self.vis_processor(vid_obj)

        return {"video": video, "index": index}
