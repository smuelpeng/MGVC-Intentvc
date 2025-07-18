import torch
import torch.nn as nn
import sklearn.metrics as metrics
import torch.nn.functional as F
import json
import os
import pandas as pd
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from collections import defaultdict
from core.ret.tvr import TVR


def tokenize(text):
    """Tokenize text into words."""
    return nltk.word_tokenize(text.lower())


def calculate_bleu_similarity(gt_captions, candidate_caption):
    candidate_caption = tokenize(candidate_caption)
    gt_captions = [tokenize(caption) for caption in gt_captions]

    bleu = sentence_bleu(
        gt_captions,
        candidate_caption,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1
    )
    return bleu


def load_captions(caption_path):
    with open(caption_path, 'r') as f:
        captions = json.load(f)['captions']
    # remove empty captions
    new_captions = {}
    for video_id, captions in captions.items():
        for caption in captions:
            if len(caption) > 0:
                new_captions.setdefault(video_id, []).append(caption)
    return new_captions


class TVRScorer():
    def __init__(self, config_path):
        self.tvr_engine = TVR(config_path)

    def score(self, video_path, captions):
        itm_scores = []
        for caption in captions:
            itm_scores.append(self.tvr_engine.compute_similarity(
                video_path, caption, method="itm"))
        return itm_scores


class BleuScorer():
    def __init__(self):
        pass

    def score(self, captions):
        bleu_scores = []
        # captions = [tokenize(caption) for caption in captions]
        for i, caption in enumerate(captions):
            other_captions = captions.copy()
            other_captions.remove(caption)
            other_captions = list(set(other_captions))
            if len(other_captions) > 0:
                bleu_scores.append(calculate_bleu_similarity(
                    other_captions, caption))
            else:
                bleu_scores.append(1)
            
        return bleu_scores


class RAGScorer:
    def __init__(self, test_caption_path, train_caption_path, video_root='data/intentvc', config_path='checkpoint/config.yaml', topk=10):
        self.test_caption_path = test_caption_path
        self.train_caption_path = train_caption_path
        self.video_root = video_root
        self.topk = topk
        self.tvr_engine = TVR(config_path)

        self.test_captions = load_captions(test_caption_path)
        self.test_phase = os.path.basename(test_caption_path).split('.')[0]
        self.db_captions = load_captions(train_caption_path)

        # Organize captions and video_ids by category
        self.class_captions = defaultdict(list)  # {category: [caption, ...]}
        self.class_video_ids = defaultdict(list)  # {category: [video_id, ...]}
        for video_id, captions in self.db_captions.items():
            category = video_id.split('-')[0]
            self.class_captions[category].extend(captions)
            self.class_video_ids[category].extend([video_id] * len(captions))

        self.sim_path = f"models/RAG_similarity.pth"
        self._prepare_video_similarity()

    def get_text_features(self, captions):
        """
        Get text features for each caption.
        """
        text_features = []
        for caption in captions:
            text_feat = self.tvr_engine.get_text_embedding(caption).cpu().numpy()
            text_features.append(text_feat)
        return text_features

    def _prepare_video_similarity(self):
        """
        """
        data = torch.load(self.sim_path)
        self.video_db_similarity = data["video_db_similarity"]
        self.video_db_similarity_topk_ids = data["video_db_similarity_topk_ids"]

    def score(self, captions, test_video_id):
        """
        Compute rag_score for each caption.
        Args:
            captions: list[str], captions to be evaluated
            test_video_id: str, current test video id
        Returns:
            rag_scores: list[float], rag score for each caption
        """
        rag_scores = []
        category = test_video_id.split('-')[0]
        db_captions = self.class_captions[category]
        db_video_ids = self.class_video_ids[category]
        topk_db_video_ids = self.video_db_similarity_topk_ids.get(
            test_video_id, set())
        for caption in captions:
            # Compute BLEU score with all training captions
            bleu_scores = [calculate_bleu_similarity(
                [db_caption], caption) for db_caption in db_captions]
            # Get top-k training video_ids with highest BLEU score
            topk_idxs = np.argsort(bleu_scores)[-self.topk:]
            topk_bleu_video_ids = [db_video_ids[i] for i in topk_idxs]
            # Count how many of the top-k BLEU video_ids are in the top-k similarity video_ids
            count = sum(
                vid in topk_db_video_ids for vid in topk_bleu_video_ids)
            rag_scores.append(count / self.topk)
        return rag_scores


if __name__ == "__main__":
    # Mock data for testing
    test_captions = {
        "cat-01": ["A cat is playing with a ball.", "A kitten is playing."],
        "dog-02": ["A dog is running in the park.", "A puppy is running."]
    }

    # Test BleuScorer
    bleu_scorer = BleuScorer()
    bleu_scores = bleu_scorer.score(
        [
            "white cat interacts with black cat on a concrete surface near greenery",
            "white cat interacts with black cat on the ground while exploring surroundings",
            "white cat moves quickly across the ground while interacting with another cat nearby",
            "white cat interacts with black cat while standing on the ground near plants",
            "white cat walks on the ground while black cat stands nearby observing",
            "white cat walks on the ground near another cat in an outdoor setting",
            "white cat walks on the ground while interacting with another cat nearby",
        ])
    print("BleuScorer scores:", bleu_scores)

    try:
        tvr_scorer = TVRScorer(config_path="models/checkpoint/config.yaml")
        video_path = "data/intentvc/videos_maxbbox/cat-1.mp4"
        tvr_scores = tvr_scorer.score(video_path, [
            "a brown cat is walking across the grass while observing nearby birds",
            "brown cat walking through grass in the presence of nearby birds on ground",
            "brown cat is walking across the green grass while black birds are nearby",
            "brown cat moves across the grass while black birds are in the background",
            "brown cat walking through green grass while observing nearby dark objects",
            "black striated cat walking on the floor near a cabinet and a wall",
            "black striated cat moving across the floor towards a corner object",
            "black striated cat walking across the floor near furniture and an object in the distance",
            "the black striated cat walks across the tiled floor towards an unseen object",
            "black striated cat walking on a tiled floor near furniture and a dark object"
        ])
        print("TVRScorer scores:", tvr_scores)
    except Exception as e:
        print("TVRScorer test skipped (TVR not implemented):", e)

    try:
        rag_scorer = RAGScorer(test_caption_path="data/sample_result_public.json",
                               train_caption_path="data/train.json",
                               video_root="data/intentvc/videos_maxbbox",
                               config_path="models/checkpoint/config.yaml", topk=10)

        predict_test_public_path = "data/result_public.json"
        predict_test_public = json.load(open(predict_test_public_path, "r"))['captions']
        for video_id, captions in predict_test_public.items():
            rag_scores = rag_scorer.score(captions, video_id)
            for caption, rag_score in zip(captions, rag_scores):
                print(video_id, caption, rag_score)
    except Exception as e:
        print("RAGScorer test skipped (TVR not implemented):", e)
