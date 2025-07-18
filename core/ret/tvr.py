
import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from typing import List, Union, Dict, Any
from pathlib import Path
from functools import lru_cache
import hashlib
from collections import OrderedDict

from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.models.blip_video_post_models.blip_video_post_retrieval import BlipVideoPostRetrieval

# Import for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.tasks import *


os.environ['HF_HOME'] = '/mnt/pfs/share/pretrained_model/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/pfs/users/yuzhipeng/workspace/Video/RTQ-MM2023/modelzoo'


class TVR:
    def __init__(self, cfg_path: str = "checkpoint/config.yaml"):
        """
        Initialize the video-text retrieval model.
        Args:
            cfg_path: Path to the config file.
        """
        class Args:
            def __init__(self, cfg_path):
                self.cfg_path = cfg_path
                self.options = []
        
        args = Args(cfg_path)
        self.cfg = Config(args)
        self.device = torch.device(
            self.cfg.run_cfg.device if torch.cuda.is_available() else "cpu")

        self.model = None
        self.video_processor = None
        self.text_processor = None

        # Initialize cache for video outputs using OrderedDict for LRU behavior
        self._video_cache = OrderedDict()
        self._cache_max_size = 10000
        self._cache_hits = 0
        self._cache_misses = 0

        self.init_model()
        self.init_processors()

    def _get_video_hash(self, video_path: str) -> str:
        """
        Generate a hash for the video path to use as cache key.
        Args:
            video_path: Path to the video file.
        Returns:
            Hash string for the video path.
        """
        return hashlib.md5(video_path.encode()).hexdigest()

    def _get_cached_video_outputs(self, video_path: str) -> Union[tuple, None]:
        """
        Get cached video outputs if available.
        Args:
            video_path: Path to the video file.
        Returns:
            Cached video outputs (video_feat, video_outputs) or None if not cached.
        """
        video_hash = self._get_video_hash(video_path)
        if video_hash in self._video_cache:
            # Move to end to mark as recently used (LRU behavior)
            result = self._video_cache[video_hash]
            self._video_cache.move_to_end(video_hash)
            self._cache_hits += 1
            return result
        self._cache_misses += 1
        return None

    def _cache_video_outputs(self, video_path: str, video_feat: torch.Tensor, video_outputs: torch.Tensor):
        """
        Cache video outputs.
        Args:
            video_path: Path to the video file.
            video_feat: Video features tensor.
            video_outputs: Video outputs tensor.
        """
        video_hash = self._get_video_hash(video_path)
        
        # Remove if already exists (will be re-added at the end)
        if video_hash in self._video_cache:
            del self._video_cache[video_hash]
        
        # Implement LRU cache behavior
        if len(self._video_cache) >= self._cache_max_size:
            # Remove the oldest entry (first item in OrderedDict)
            self._video_cache.popitem(last=False)
        
        # Cache the video outputs (adds to end of OrderedDict)
        self._video_cache[video_hash] = (video_feat, video_outputs)

    def _get_video_features_with_cache(self, video_path: str) -> tuple:
        """
        Get video features with caching support.
        Args:
            video_path: Path to the video file.
        Returns:
            Tuple of (video_feat, video_outputs).
        """
        # Check cache first
        cached_result = self._get_cached_video_outputs(video_path)
        if cached_result is not None:
            return cached_result
        
        # If not in cache, compute and cache
        video_tensor = self.preprocess_video(video_path)
        video_feat, video_outputs = self.model.visual_encoder.forward_features(video_tensor)
        
        # Cache the results
        self._cache_video_outputs(video_path, video_feat, video_outputs)
        
        return video_feat, video_outputs

    def clear_cache(self):
        """Clear the video cache and reset statistics."""
        self._video_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def set_cache_size(self, max_size: int):
        """
        Set the maximum cache size.
        Args:
            max_size: Maximum number of items to cache.
        """
        if max_size < 0:
            raise ValueError("Cache size must be non-negative")
        
        self._cache_max_size = max_size
        
        # If new size is smaller, remove oldest entries
        while len(self._video_cache) > self._cache_max_size:
            self._video_cache.popitem(last=False)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        Returns:
            Dictionary with cache statistics.
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._video_cache),
            'max_cache_size': self._cache_max_size,
            'cache_usage_percentage': len(self._video_cache) / self._cache_max_size * 100,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percentage': hit_rate
        }

    def init_model(self):
        """Initialize the model from config."""
        model_cfg = self.cfg.model_cfg
        self.model = BlipVideoPostRetrieval.from_config(model_cfg)

        # Load checkpoint if provided
        if hasattr(self.cfg, 'resume_ckpt_path') and self.cfg.resume_ckpt_path:
            checkpoint = torch.load(
                self.cfg.resume_ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def init_processors(self):
        """Initialize video and text processors from config."""
        vis_cfg = self.cfg.datasets_cfg.intentvc_retrieval.vis_processor.eval
        self.video_processor = registry.get_processor_class(
            vis_cfg.name).from_config(vis_cfg)

        text_cfg = self.cfg.datasets_cfg.intentvc_retrieval.text_processor.eval
        self.text_processor = registry.get_processor_class(
            text_cfg.name).from_config(text_cfg)

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess a video file into a tensor suitable for the model.
        Args:
            video_path: Path to the video file.
        Returns:
            Preprocessed video tensor.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_tensor = self.video_processor(video_path)
        if len(video_tensor.shape) == 4:
            video_tensor = video_tensor.unsqueeze(0)
        return video_tensor.to(self.device)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a text string for the model.
        Args:
            text: Input text.
        Returns:
            Preprocessed text.
        """
        return self.text_processor(text)

    def get_video_embedding(self, video_path: str) -> torch.Tensor:
        """
        Get the embedding vector for a video.
        Args:
            video_path: Path to the video file.
        Returns:
            Video embedding tensor.
        """
        with torch.no_grad():
            video_feat, _ = self._get_video_features_with_cache(video_path)
            # Handle different tensor shapes after vision projection
            if len(video_feat.shape) == 3:
                video_feat = self.model.vision_proj(video_feat[:, 0, :])  # Use the first token
            else:
                video_feat = self.model.vision_proj(video_feat)
            video_feat = torch.nn.functional.normalize(video_feat, dim=-1)
            return video_feat

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding vector for a text string.
        Args:
            text: Input text.
        Returns:
            Text embedding tensor.
        """
        with torch.no_grad():
            processed_text = self.preprocess_text(text)
            text_tokens = self.model.tokenizer(
                processed_text,
                padding="max_length",
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            text_output = self.model.text_encoder.forward_text(text_tokens)
            text_embeds = text_output.last_hidden_state
            # Handle different tensor shapes after text projection
            if len(text_embeds.shape) == 3:
                text_feat = self.model.text_proj(text_embeds[:, 0, :])
            else:
                text_feat = self.model.text_proj(text_embeds)
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
            return text_feat

    def compute_similarity(self, video_path: str, text: str, method: str = "cosine") -> float:
        """
        Compute similarity between a video and a text using the specified method.
        Args:
            video_path: Path to the video file.
            text: Text description.
            method: Similarity method ("cosine", "itm", "both").
        Returns:
            Similarity score (0-1).
        """
        if method == "cosine":
            return self._compute_cosine_similarity(video_path, text)
        elif method == "itm":
            return self._compute_itm_similarity(video_path, text)
        elif method == "both":
            cosine_sim = self._compute_cosine_similarity(video_path, text)
            itm_sim = self._compute_itm_similarity(video_path, text)
            return (cosine_sim + itm_sim) / 2
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

    def _compute_cosine_similarity(self, video_path: str, text: str) -> float:
        """Compute cosine similarity between video and text embeddings."""
        video_embedding = self.get_video_embedding(video_path)
        text_embedding = self.get_text_embedding(text)
        similarity = F.cosine_similarity(
            video_embedding, text_embedding, dim=-1)
        similarity = (similarity + 1) / 2
        return similarity.item()

    def _compute_itm_similarity(self, video_path: str, text: str) -> float:
        """Compute similarity using the ITM head (matching probability)."""
        with torch.no_grad():
            processed_text = self.preprocess_text(text)
            video_feat, video_outputs = self._get_video_features_with_cache(video_path)
            video_atts = torch.ones(video_outputs.size()[
                                    :-1], dtype=torch.long).to(self.device)
            text_tokens = self.model.tokenizer(
                processed_text,
                padding="max_length",
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            encoder_input_ids = text_tokens.input_ids.clone()
            encoder_input_ids[:, 0] = self.model.tokenizer.enc_token_id
            output = self.model.text_encoder(
                encoder_input_ids,
                attention_mask=text_tokens.attention_mask,
                encoder_hidden_states=video_outputs,
                encoder_attention_mask=video_atts,
                return_dict=True,
            )
            vl_embeddings = output.last_hidden_state[:, 0, :]
            itm_logits = self.model.itm_head(vl_embeddings)
            itm_probs = F.softmax(itm_logits, dim=-1)
            itm_similarity = itm_probs[:, 1].item()
            return itm_similarity

    def get_sim_matrix(self, captions: List[str], video_paths: List[str], method: str = "cosine") -> np.ndarray:
        """
        Compute a similarity matrix for multiple texts and videos.
        Args:
            captions: List of text descriptions.
            video_paths: List of video file paths.
            method: Similarity method ("cosine", "itm", "both").
        Returns:
            Similarity matrix (num_texts, num_videos).
        """
        num_texts = len(captions)
        num_videos = len(video_paths)
        sim_matrix = np.zeros((num_texts, num_videos))
        for i, caption in enumerate(captions):
            for j, video_path in enumerate(video_paths):
                try:
                    sim_matrix[i, j] = self.compute_similarity(
                        video_path, caption, method)
                except Exception as e:
                    print(
                        f"Error computing similarity for caption {i} and video {j}: {e}")
                    sim_matrix[i, j] = 0.0
        return sim_matrix

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data dict for model input.
        Args:
            data: Dict containing 'video_path' and/or 'text'.
        Returns:
            Dict with processed tensors.
        """
        processed_data = {}
        if 'video_path' in data:
            processed_data['video_tensor'] = self.preprocess_video(
                data['video_path'])
        if 'text' in data:
            processed_data['text'] = self.preprocess_text(data['text'])
        return processed_data

    def batch_compute_similarity(self, video_paths: List[str], texts: List[str], method: str = "cosine") -> List[float]:
        """
        Compute similarity for batches of videos and texts.
        Args:
            video_paths: List of video file paths.
            texts: List of text descriptions.
            method: Similarity method ("cosine", "itm", "both").
        Returns:
            List of similarity scores.
        """
        if len(video_paths) != len(texts):
            raise ValueError("video_paths and texts must have the same length")
        similarities = []
        for video_path, text in zip(video_paths, texts):
            try:
                sim = self.compute_similarity(
                    video_path, text, method)
                similarities.append(sim)
            except Exception as e:
                print(f"Error computing similarity for {video_path}: {e}")
                similarities.append(0.0)
        return similarities

if __name__ == "__main__":
    import sys
    import numpy as np
    print("=== TVR API Test ===")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    tvr = TVR("checkpoint/config.yaml")
    # Test data (replace with actual video path)
    video_path = "demo/example1.mp4"
    video_paths = [video_path]
    texts = [
        "A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
        "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
        "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
        "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
        "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
        "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
        "A playful dog slides down a snowy hill, wagging its tail with delight.",
        "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
        "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
        "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]

    for method in ["cosine", "itm", "both"]:
        try:
            if os.path.exists(video_path):
                sim = tvr.compute_similarity(
                    video_path, texts[0], method=method)
                print(f"{method} similarity: {sim:.4f}")
            else:
                print(f"Video file not found: {video_path}")
        except Exception as e:
            print(f"{method} similarity error: {e}")

    print("\n[2] Batch similarity (cosine/itm)")
    existing_videos = [vp for vp in video_paths if os.path.exists(vp)]
    existing_texts = texts[:len(existing_videos)]
    if existing_videos:
        for method in ["cosine", "itm"]:
            try:
                sims = tvr.batch_compute_similarity(
                    existing_videos, existing_texts, method=method)
                print(f"{method} batch similarity: {sims}")
            except Exception as e:
                print(f"{method} batch similarity error: {e}")
    else:
        print("No valid video files, skipping batch test.")

    print("\n[3] Similarity matrix (cosine/itm)")
    if existing_videos:
        for method in ["cosine", "itm"]:
            try:
                sim_matrix = tvr.get_sim_matrix(
                    texts, existing_videos, method=method)
                print(f"{method} similarity matrix:\n{sim_matrix}")
            except Exception as e:
                print(f"{method} similarity matrix error: {e}")
    else:
        print("No valid video files, skipping matrix test.")

    print("\n[4] Embedding shape output")
    try:
        if os.path.exists(video_path):
            v_emb = tvr.get_video_embedding(video_path)
            print(f"Video embedding shape: {v_emb.shape}")
        t_emb = tvr.get_text_embedding(texts[0])
        print(f"Text embedding shape: {t_emb.shape}")
    except Exception as e:
        print(f"Embedding shape test error: {e}")