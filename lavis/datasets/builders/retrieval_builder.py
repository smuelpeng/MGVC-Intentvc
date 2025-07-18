"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.retrieval_datasets import (
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)

from lavis.datasets.datasets.intentvc_video_retrieval_datasets import (
    IntentVCVideoRetrievalDataset,
    IntentVCVideoRetrievalEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults_ret.yaml"}

@registry.register_builder("intentvc_retrieval")
class IntentVCRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = IntentVCVideoRetrievalDataset
    eval_dataset_cls = IntentVCVideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/intentvc/defaults_ret.yaml"}
