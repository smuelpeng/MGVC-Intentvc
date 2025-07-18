import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from decord import VideoReader
from tqdm import tqdm


class VideoAugmentation:
    """Video augmentation class for processing videos with bounding boxes."""
    
    def __init__(self, min_frames: int = 12):
        """
        Initialize VideoAugmentation.
        
        Args:
            min_frames: Minimum number of frames to keep
        """
        self.min_frames = min_frames
    
    def read_bbox_file(self, bbox_path: str) -> List[List[int]]:
        """
        Read bounding box file.
        
        Args:
            bbox_path: Path to bbox file, each line format: x1,y1,w,h
            
        Returns:
            List of bounding boxes as [x1, y1, w, h]
        """
        with open(bbox_path, 'r') as f:
            bboxes = [list(map(int, line.strip().split(','))) for line in f.readlines()]
        return bboxes
    
    def _load_video_and_bboxes(self, video_path: str, bbox_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Load video frames and bounding boxes, filtering out invalid ones.
        
        Args:
            video_path: Path to video file
            bbox_path: Path to bounding box file
            
        Returns:
            Tuple of (frames, bboxes) with invalid entries filtered out
        """
        # Load video
        video_reader = VideoReader(video_path)
        frames = video_reader.get_batch(range(len(video_reader))).asnumpy()
        frames_list = [frames[i] for i in range(len(frames))]
        
        # Load bounding boxes
        bboxes = self.read_bbox_file(bbox_path)
        
        print(f"Got {len(frames)} frames and {len(bboxes)} bboxes shape {frames[0].shape}")
        
        # Filter valid bboxes (w > 0 and h > 0)
        valid_frames = []
        valid_bboxes = []
        
        for frame, bbox in zip(frames_list, bboxes):
            x1, y1, w, h = bbox
            if w > 0 and h > 0:
                valid_frames.append(frame)
                valid_bboxes.append(bbox)
            else:
                print(f"Bad bbox: {bbox}")
        
        return valid_frames, valid_bboxes
    
    def _draw_bbox_on_frame(self, frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, w, h]
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Frame with drawn bounding box
        """
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        return cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    def _adjust_bbox_coordinates(self, bbox: List[int], crop_x1: int, crop_y1: int) -> List[int]:
        """
        Adjust bbox coordinates relative to crop region.
        
        Args:
            bbox: Original bbox [x1, y1, w, h]
            crop_x1: Crop region x1
            crop_y1: Crop region y1
            
        Returns:
            Adjusted bbox [x1, y1, x2, y2]
        """
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        new_x1 = x1 - crop_x1
        new_y1 = y1 - crop_y1
        new_x2 = x2 - crop_x1
        new_y2 = y2 - crop_y1
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def _validate_crop_region(self, crop_x1: int, crop_y1: int, crop_x2: int, crop_y2: int) -> bool:
        """
        Validate crop region coordinates.
        
        Args:
            crop_x1, crop_y1, crop_x2, crop_y2: Crop region coordinates
            
        Returns:
            True if valid, False otherwise
        """
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            print(f"Invalid crop region: x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
            return False
        return True
    
    def get_crop_video_max_bound_bbox(self, video_path: str, bbox_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Crop video to contain all bounding boxes within a maximum boundary.
        
        Args:
            video_path: Path to video file
            bbox_path: Path to bounding box file
            
        Returns:
            Tuple of (cropped_frames, cropped_bboxes)
        """
        frames, bboxes = self._load_video_and_bboxes(video_path, bbox_path)
        
        if not frames or not bboxes:
            return [], []
        
        # Convert bboxes to x1,y1,x2,y2 format
        bboxes_x1y1x2y2 = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            bboxes_x1y1x2y2.append([x1, y1, x1 + w, y1 + h])
        
        bboxes_array = np.array(bboxes_x1y1x2y2)
        
        # Calculate maximum boundary
        min_x1, min_y1 = bboxes_array[:, 0].min(), bboxes_array[:, 1].min()
        max_x2, max_y2 = bboxes_array[:, 2].max(), bboxes_array[:, 3].max()
        
        # Crop video
        cropped_frames = []
        cropped_bboxes = []
        
        for frame, bbox in zip(frames, bboxes):
            # Draw bbox on frame
            frame = self._draw_bbox_on_frame(frame, bbox)
            
            # Calculate crop region
            frame_height, frame_width = frame.shape[:2]
            crop_x1 = max(0, min_x1)
            crop_y1 = max(0, min_y1)
            crop_x2 = min(frame_width, max_x2)
            crop_y2 = min(frame_height, max_y2)
            
            # Validate crop region
            if not self._validate_crop_region(crop_x1, crop_y1, crop_x2, crop_y2):
                continue
            
            # Crop frame
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped_frame.size == 0:
                print(f"Empty cropped frame for bbox: {bbox}")
                continue
            
            cropped_frames.append(cropped_frame)
            cropped_bboxes.append(self._adjust_bbox_coordinates(bbox, crop_x1, crop_y1))
        
        return cropped_frames, cropped_bboxes
    
    def get_cropped_video_center_bbox(self, video_path: str, bbox_path: str, scale: float = 2.0) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Crop video centered on bounding boxes with scaling.
        
        Args:
            video_path: Path to video file
            bbox_path: Path to bounding box file
            scale: Scale factor for crop size
        
        Returns:
            Tuple of (cropped_frames, cropped_bboxes)
        """
        frames, bboxes = self._load_video_and_bboxes(video_path, bbox_path)
        
        if not frames or not bboxes:
            return [], []
        
        # Calculate maximum dimensions
        max_w = int(max(bbox[2] for bbox in bboxes) * scale)
        max_h = int(max(bbox[3] for bbox in bboxes) * scale)
        
        cropped_frames = []
        cropped_bboxes = []
        
        for frame, bbox in zip(frames, bboxes):
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # Draw bbox on frame
            frame = self._draw_bbox_on_frame(frame, bbox)
            
            # Calculate bbox center
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            
            # Calculate crop region
            crop_x1 = bbox_center_x - max_w // 2
            crop_y1 = bbox_center_y - max_h // 2
            crop_x2 = crop_x1 + max_w
            crop_y2 = crop_y1 + max_h
            
            frame_height, frame_width = frame.shape[:2]
            
            # Adjust crop region to stay within frame boundaries
            if crop_x1 < 0:
                crop_x2 += -crop_x1
                crop_x1 = 0
            if crop_y1 < 0:
                crop_y2 += -crop_y1
                crop_y1 = 0
            if crop_x2 > frame_width:
                crop_x1 -= (crop_x2 - frame_width)
                crop_x2 = frame_width
            if crop_y2 > frame_height:
                crop_y1 -= (crop_y2 - frame_height)
                crop_y2 = frame_height
            # 防止 crop_x1/crop_y1 变负
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            # 转 int
            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, [crop_x1, crop_y1, crop_x2, crop_y2])
            # 检查 crop 区域宽高
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                print(f"Invalid crop region: {crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}")
                continue
            # Crop frame
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_frame.size == 0:
                print(f"Empty cropped frame for bbox: {bbox}")
                continue
            # Adjust bbox coordinates
            new_bbox = self._adjust_bbox_coordinates(bbox, crop_x1, crop_y1)
            # Ensure bbox coordinates are within cropped frame bounds
            new_bbox[0] = max(0, min(new_bbox[0], cropped_frame.shape[1] - 1))
            new_bbox[1] = max(0, min(new_bbox[1], cropped_frame.shape[0] - 1))
            new_bbox[2] = max(0, min(new_bbox[2], cropped_frame.shape[1]))
            new_bbox[3] = max(0, min(new_bbox[3], cropped_frame.shape[0]))
            cropped_frames.append(cropped_frame)
            cropped_bboxes.append(new_bbox)
        return cropped_frames, cropped_bboxes
    
    def get_basic_video_aug(self, video_path: str, bbox_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Basic video augmentation - draw bounding boxes on frames.
        
        Args:
            video_path: Path to video file
            bbox_path: Path to bounding box file
            
        Returns:
            Tuple of (frames_with_bboxes, bboxes)
        """
        frames, bboxes = self._load_video_and_bboxes(video_path, bbox_path)
        
        if not frames or not bboxes:
            return [], []
        
        processed_frames = []
        for frame, bbox in zip(frames, bboxes):
            frame = self._draw_bbox_on_frame(frame, bbox)
            processed_frames.append(frame)
        
        return processed_frames, bboxes
        


        