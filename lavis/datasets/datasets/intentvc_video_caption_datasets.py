"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import io
import json
from mmengine.fileio import FileClient 



from lavis.datasets.datasets.base_dataset import BaseDataset


class IntentVCVideoCaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # # Forge annotations
        # self.annotation = reforge_annotations(self.annotation)

        # # Explode annotations for training
        # self.annotation = explode_annotations(self.annotation)
        self.annotation = []
        self.vis_root = vis_root


        self.img_ids = {}
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        # self.ann_paths = ann_paths

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
            vid_obj = io.BytesIO(open(ann["video"], 'rb').read())

        video = self.vis_processor(vid_obj)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class IntentVCVideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # # Forge annotations
        # self.annotation = reforge_annotations(self.annotation)

        # # Get file client
        # self.file_client = FileClient(backend='lmdb', db_path=vis_root)
        self.vis_processor = vis_processor
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

            for i, caption in enumerate(v):
                self.annotation.append({
                    "video": f'{vis_root}/{category}/{video_id}/{video_id}.mp4',
                    "caption": caption,
                    "instance_id": anno_num,
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
            vid_obj = io.BytesIO(open(ann["video"], 'rb').read())

        video = self.vis_processor(vid_obj)

        return {
            "video": video,
            "image_id": ann["video"],
            "instance_id": ann["instance_id"],
        }
