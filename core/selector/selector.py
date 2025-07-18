import json
from tqdm import tqdm
from dataclasses import dataclass
from core.ret.tvr import TVR
from core.selector.scorer import TVRScorer, BleuScorer, RAGScorer
from core.selector.scorer import tokenize, calculate_bleu_similarity, load_captions
from core.cap.prompt_util import ALL_CATEGORIES
import numpy as np
from utils.config import load_config

from sklearn.linear_model import LinearRegression
import torch

class CaptionSelector:
    @dataclass
    class Config:
        tvr_path: str = "models/checkpoint/config.yaml"
        test_caption_path: str = "data/sample_result_public.json"
        train_caption_path: str = "data/train.json"
        video_root: str = "data/intentvc/videos_maxbbox"
        topk: int = 10

    config: Config
    def __init__(self, config: Config):
        self.tvr_scorer = TVRScorer(config.tvr_path)
        self.bleu_scorer = BleuScorer()
        self.rag_scorer = RAGScorer(
            config.test_caption_path, 
            config.train_caption_path, 
            config.video_root, 
            config.tvr_path,
            config.topk)
        self.train_db_captions = load_captions(config.train_caption_path)

        self.reg_modes = {}
        self.cfg = config
        self.video_root = config.video_root

        test_tag = os.path.basename(self.cfg.test_caption_path).split('.')[0]
        train_tag = os.path.basename(self.cfg.train_caption_path).split('.')[0]
        self.reg_model_path = f"models/reg_model.pth"
        data = torch.load(self.reg_model_path)
        self.reg_modes = data["reg_modes"]
    

    def select(self, video_id, captions, topk=20):
        captions = list(set(captions))
        video_path = os.path.join(self.video_root, video_id + ".mp4")
        tvr_scores = self.tvr_scorer.score(video_path, captions)
        bleu_scores = self.bleu_scorer.score(captions)
        rag_scores = self.rag_scorer.score(captions, video_id)

        # 转为numpy数组以支持高级索引
        tvr_scores = np.array(tvr_scores)
        bleu_scores = np.array(bleu_scores)
        rag_scores = np.array(rag_scores)

        # select top20 tvr scores for rerank
        topk_tvr_scores = np.argsort(tvr_scores)[::-1][:topk]
        topk_captions = [captions[i] for i in topk_tvr_scores]

        tvr_scores = tvr_scores[topk_tvr_scores]
        bleu_scores = bleu_scores[topk_tvr_scores]
        rag_scores = rag_scores[topk_tvr_scores]

        # rerank    
        text_features = self.rag_scorer.get_text_features(topk_captions)
        text_features = np.concatenate(text_features, axis=0)

        category = video_id.split('-')[0]
        reg_mode = self.reg_modes[category]
        scores = np.array([tvr_scores, bleu_scores, rag_scores])
        scores = scores.T

        text_features = np.concatenate([text_features, scores], axis=-1)
        final_scores = reg_mode.predict(text_features)

        # select max score caption
        max_score_idx = np.argmax(final_scores)
        return topk_captions[max_score_idx]


if __name__ == "__main__":
    import os
    import json

    test_caption_path = "data/result_public.55.json"
    train_caption_path = "data/train.json"
    # 配置
    config = CaptionSelector.Config(
        tvr_path="models/checkpoint/config.yaml",  # 需存在或可用 dummy 文件
        test_caption_path=test_caption_path,
        train_caption_path=train_caption_path,
        video_root="data/intentvc/videos_maxbbox",
        topk=10
    )
    selector = CaptionSelector(config)
    print("[TEST] Selecting...")

    public_result_path = "data/result_public.55.json"
    public_result = load_captions(public_result_path)
    result_selected = {}
    for video_id, captions in tqdm(public_result.items(), total=len(public_result)):
        result = selector.select(video_id, captions[5:])
        # print(video_id, result)
        result_selected[video_id] = result

    with open("data/result_public_selected.json", "w") as f:
        json.dump({
            'version': 'v1',
            "captions": result_selected
        }, f, indent=2)