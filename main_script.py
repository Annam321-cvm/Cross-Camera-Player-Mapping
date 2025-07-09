import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import time
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import threading
from queue import Queue
import sys

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PlayerDetection:
    """Data class for player detection information"""
    player_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    features: np.ndarray
    timestamp: float
    camera_id: int
    center: Tuple[int, int]
    frame_number: int  # Added for better frame tracking

class FeatureExtractor:
    """Extract visual and spatial features from player detections"""
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        self.setup_feature_extractor()
    
    def setup_feature_extractor(self):
        """Setup CNN feature extractor for appearance features"""
        logger.info("Setting up feature extractor...")
        # Using ResNet-like feature extractor
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.feature_dim)
        )
        
        # Initialize weights
        for m in self.feature_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
        logger.info("Feature extractor setup complete")
    
    def extract_appearance_features(self, img_crop: np.ndarray) -> np.ndarray:
        """Extract appearance features from player crop"""
        if img_crop.size == 0:
            return np.zeros(self.feature_dim)
            
        # Resize to standard size
        img_resized = cv2.resize(img_crop, (64, 128))
        img_tensor = torch.FloatTensor(img_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            features = self.feature_net(img_tensor)
        
        return features.squeeze().numpy()
    
    def extract_spatial_features(self, bbox: Tuple[int, int, int, int], 
                               frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract spatial features from bounding box"""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape
        
        # Normalize coordinates
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        aspect_ratio = width / height if height > 0 else 1.0
        
        return np.array([center_x, center_y, width, height, aspect_ratio])
    
    def extract_combined_features(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract combined appearance and spatial features"""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid crop coordinates
        x1 = max(0, min(x1, img.shape[1]))
        y1 = max(0, min(y1, img.shape[0]))
        x2 = max(x1, min(x2, img.shape[1]))
        y2 = max(y1, min(y2, img.shape[0]))
        
        # Extract appearance features
        img_crop = img[y1:y2, x1:x2]
        if img_crop.size > 0:
            appearance_features = self.extract_appearance_features(img_crop)
        else:
            appearance_features = np.zeros(self.feature_dim)
        
        # Extract spatial features
        spatial_features = self.extract_spatial_features(bbox, img.shape[:2])
        
        # Combine features
        combined_features = np.concatenate([appearance_features, spatial_features])
        
        return combined_features

class PlayerTracker:
    """Track players across cameras and time"""
    
    def __init__(self, max_disappear_frames=30, similarity_threshold=0.7):
        self.max_disappear_frames = max_disappear_frames
        self.similarity_threshold = similarity_threshold
        self.next_id = 1
        self.active_players = {}  # player_id -> PlayerDetection
        self.disappeared_players = {}  # player_id -> frame_count
        self.player_history = defaultdict(lambda: deque(maxlen=20))  # Increased history
        self.frame_buffer = deque(maxlen=5)  # Buffer for temporal smoothing
        logger.info(f"Player tracker initialized with similarity threshold: {similarity_threshold}")
        
    def update_player_history(self, player_id: int, detection: PlayerDetection):
        """Update player movement history"""
        self.player_history[player_id].append({
            'center': detection.center,
            'timestamp': detection.timestamp,
            'camera_id': detection.camera_id,
            'frame_number': detection.frame_number,
            'bbox': detection.bbox
        })
    
    def calculate_motion_features(self, player_id: int) -> np.ndarray:
        """Calculate motion-based features from player history"""
        if len(self.player_history[player_id]) < 2:
            return np.zeros(3)  # velocity_x, velocity_y, acceleration
        
        history = list(self.player_history[player_id])
        
        # Calculate velocity
        recent_pos = history[-1]['center']
        prev_pos = history[-2]['center']
        time_diff = history[-1]['timestamp'] - history[-2]['timestamp']
        
        if time_diff > 0:
            velocity_x = (recent_pos[0] - prev_pos[0]) / time_diff
            velocity_y = (recent_pos[1] - prev_pos[1]) / time_diff
        else:
            velocity_x = velocity_y = 0
        
        # Calculate acceleration if we have enough history
        if len(history) >= 3:
            prev_vel_x = (prev_pos[0] - history[-3]['center'][0]) / max(history[-2]['timestamp'] - history[-3]['timestamp'], 1e-6)
            acceleration = abs(velocity_x - prev_vel_x)
        else:
            acceleration = 0
        
        return np.array([velocity_x, velocity_y, acceleration])
    
    def find_best_match(self, new_detection: PlayerDetection) -> Optional[int]:
        """Find best matching player ID for new detection"""
        if not self.active_players:
            return None
        
        best_match_id = None
        best_similarity = 0
        
        for player_id, prev_detection in self.active_players.items():
            # Skip if from same camera (assuming single player per camera view)
            if prev_detection.camera_id == new_detection.camera_id:
                continue
            
            # Calculate feature similarity
            feature_similarity = cosine_similarity(
                new_detection.features.reshape(1, -1),
                prev_detection.features.reshape(1, -1)
            )[0][0]
            
            # Calculate temporal consistency
            time_diff = abs(new_detection.timestamp - prev_detection.timestamp)
            temporal_weight = max(0, 1 - time_diff / 5.0)  # Decay over 5 seconds
            
            # Calculate spatial consistency (if same camera)
            spatial_weight = 1.0
            if prev_detection.camera_id == new_detection.camera_id:
                distance = np.linalg.norm(np.array(new_detection.center) - np.array(prev_detection.center))
                spatial_weight = max(0, 1 - distance / 200)  # Decay over 200 pixels
            
            # Combined similarity score
            combined_similarity = feature_similarity * temporal_weight * spatial_weight
            
            if combined_similarity > best_similarity and combined_similarity > self.similarity_threshold:
                best_similarity = combined_similarity
                best_match_id = player_id
        
        return best_match_id
    
    def update(self, detections: List[PlayerDetection]) -> Dict[int, PlayerDetection]:
        """Update tracker with new detections"""
        # Mark all current players as disappeared initially
        for player_id in list(self.active_players.keys()):
            if player_id not in self.disappeared_players:
                self.disappeared_players[player_id] = 0
        
        # Process new detections
        for detection in detections:
            # Try to match with existing players
            matched_id = self.find_best_match(detection)
            
            if matched_id is not None:
                # Update existing player
                detection.player_id = matched_id
                self.active_players[matched_id] = detection
                self.update_player_history(matched_id, detection)
                
                # Remove from disappeared list
                if matched_id in self.disappeared_players:
                    del self.disappeared_players[matched_id]
            else:
                # Create new player
                detection.player_id = self.next_id
                self.active_players[self.next_id] = detection
                self.update_player_history(self.next_id, detection)
                self.next_id += 1
        
        # Remove players that have disappeared for too long
        to_remove = []
        for player_id, disappear_count in self.disappeared_players.items():
            if disappear_count >= self.max_disappear_frames:
                to_remove.append(player_id)
            else:
                self.disappeared_players[player_id] += 1
        
        for player_id in to_remove:
            if player_id in self.active_players:
                del self.active_players[player_id]
            del self.disappeared_players[player_id]
        
        return self.active_players

class PlayerReIDSystem:
    """Main Player Re-Identification System"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.setup_model()
        self.feature_extractor = FeatureExtractor()
        self.tracker = PlayerTracker()
        self.frame_count = 0
        
    def setup_model(self):
        """Setup YOLOv11 model"""
        try:
            logger.info(f"Loading YOLOv11 model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded YOLOv11 model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_players(self, frame: np.ndarray, camera_id: int, timestamp: float, frame_number: int) -> List[PlayerDetection]:
        """Detect players in frame"""
        if frame is None or frame.size == 0:
            return []
            
        results = self.model(frame, conf=self.conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Ensure valid bounding box
                    x1 = max(0, min(x1, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0]))
                    x2 = max(x1, min(x2, frame.shape[1]))
                    y2 = max(y1, min(y2, frame.shape[0]))
                    
                    if x2 - x1 < 10 or y2 - y1 < 10:  # Skip very small detections
                        continue
                    
                    # Extract features
                    features = self.feature_extractor.extract_combined_features(frame, (x1, y1, x2, y2))
                    
                    # Create detection
                    detection = PlayerDetection(
                        player_id=-1,  # Will be assigned by tracker
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        features=features,
                        timestamp=timestamp,
                        camera_id=camera_id,
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        frame_number=frame_number
                    )
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: Dict[int, PlayerDetection], 
                       camera_id: int, scale_factor: float = 1.0) -> np.ndarray:
        """Draw player detections on frame with proper scaling"""
        if frame is None or frame.size == 0:
            return frame
            
        result_frame = frame.copy()
        
        for player_id, detection in detections.items():
            if detection.camera_id != camera_id:
                continue
                
            # Scale coordinates if needed
            x1, y1, x2, y2 = detection.bbox
            if scale_factor != 1.0:
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, result_frame.shape[1]))
            y1 = max(0, min(y1, result_frame.shape[0]))
            x2 = max(x1, min(x2, result_frame.shape[1]))
            y2 = max(y1, min(y2, result_frame.shape[0]))
            
            # Use different colors for different players
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            color = colors[player_id % len(colors)]
            
            # Draw bounding box with thicker lines
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw player ID with better visibility
            label = f"Player {player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw background for text
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence
            conf_label = f"{detection.confidence:.2f}"
            cv2.putText(result_frame, conf_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(result_frame, (center_x, center_y), 5, color, -1)
        
        return result_frame
    
    def synchronize_frames(self, cap1, cap2, target_fps=30):
        """Synchronize frames from two video sources"""
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame skip ratios
        skip1 = max(1, int(fps1 / target_fps))
        skip2 = max(1, int(fps2 / target_fps))
        
        return skip1, skip2
    
    def process_videos(self, video_path1: str, video_path2: str, output_path: str = None):
        """Process two video streams for player re-identification"""
        logger.info(f"Opening video files:")
        logger.info(f"  Video 1: {video_path1}")
        logger.info(f"  Video 2: {video_path2}")
        
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        
        if not cap1.isOpened():
            logger.error(f"Error opening video file: {video_path1}")
            return
        if not cap2.isOpened():
            logger.error(f"Error opening video file: {video_path2}")
            return
        
        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate synchronization parameters
        target_fps = min(fps1, fps2, 30)  # Limit to 30 FPS for better performance
        skip1, skip2 = self.synchronize_frames(cap1, cap2, target_fps)
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            logger.info(f"Setting up video writer for output: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_height = max(height1, height2)
            output_width = width1 + width2
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (output_width, output_height))
        
        frame_count = 0
        skip_count1 = 0
        skip_count2 = 0
        start_time = time.time()
        
        logger.info(f"Starting video processing...")
        logger.info(f"Video 1: {width1}x{height1} @ {fps1} FPS")
        logger.info(f"Video 2: {width2}x{height2} @ {fps2} FPS")
        logger.info(f"Target FPS: {target_fps}")
        
        while True:
            # Read frames with synchronization
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                logger.info("End of video reached")
                break
            
            # Skip frames for synchronization
            if skip_count1 < skip1 - 1:
                skip_count1 += 1
                continue
            if skip_count2 < skip2 - 1:
                skip_count2 += 1
                continue
            
            skip_count1 = 0
            skip_count2 = 0
            
            current_time = time.time()
            
            # Detect players in both frames
            detections1 = self.detect_players(frame1, camera_id=1, timestamp=current_time, frame_number=frame_count)
            detections2 = self.detect_players(frame2, camera_id=2, timestamp=current_time, frame_number=frame_count)
            
            # Log detections
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Frame {frame_count}: Detected {len(detections1)} players in video 1, {len(detections2)} players in video 2")
            
            # Combine detections
            all_detections = detections1 + detections2
            
            # Update tracker
            tracked_players = self.tracker.update(all_detections)
            
            # Calculate scale factors for display
            scale1 = 1.0
            scale2 = 1.0
            
            # Resize frames to same height for better display
            display_height = max(height1, height2)
            if height1 != display_height:
                scale1 = display_height / height1
                frame1 = cv2.resize(frame1, (int(width1 * scale1), display_height))
            if height2 != display_height:
                scale2 = display_height / height2
                frame2 = cv2.resize(frame2, (int(width2 * scale2), display_height))
            
            # Draw results with proper scaling
            result_frame1 = self.draw_detections(frame1, tracked_players, camera_id=1, scale_factor=scale1)
            result_frame2 = self.draw_detections(frame2, tracked_players, camera_id=2, scale_factor=scale2)
            
            # Add frame information
            info_text = f"Frame: {frame_count} | Active Players: {len(tracked_players)} | FPS: {target_fps:.1f}"
            cv2.putText(result_frame1, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display results
            cv2.imshow('Camera 1 - Broadcast', result_frame1)
            cv2.imshow('Camera 2 - Tactical', result_frame2)
            
            # Save output if specified
            if output_path and out:
                # Ensure both frames have same height
                if result_frame1.shape[0] != result_frame2.shape[0]:
                    target_height = max(result_frame1.shape[0], result_frame2.shape[0])
                    if result_frame1.shape[0] < target_height:
                        result_frame1 = cv2.resize(result_frame1, (result_frame1.shape[1], target_height))
                    if result_frame2.shape[0] < target_height:
                        result_frame2 = cv2.resize(result_frame2, (result_frame2.shape[1], target_height))
                
                combined_frame = np.hstack([result_frame1, result_frame2])
                out.write(combined_frame)
            
            frame_count += 1
            
            # Print statistics every 60 frames
            if frame_count % 60 == 0:
                elapsed_time = time.time() - start_time
                actual_fps = frame_count / elapsed_time
                logger.info(f"Processed {frame_count} frames, Actual FPS: {actual_fps:.2f}, Active players: {len(tracked_players)}")
            
            # Break on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User pressed 'q' - stopping processing")
                break
            elif key == ord('p'):  # Pause on 'p'
                logger.info("User pressed 'p' - pausing (press any key to continue)")
                cv2.waitKey(0)
        
        # Cleanup
        cap1.release()
        cap2.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Processing complete. Total frames: {frame_count}")

def main():
    """Main function to run the player re-identification system"""
    print("=== Player Re-Identification System Starting ===")
    
    parser = argparse.ArgumentParser(description='Player Re-Identification System for Football Analytics')
    parser.add_argument('--model', required=True, help='Path to YOLOv11 model (.pt file)')
    parser.add_argument('--broadcast', default='broadcast.mp4', help='Path to broadcast camera video')
    parser.add_argument('--tacticam', default='tacticam.mp4', help='Path to tactical camera video')
    parser.add_argument('--output', default='football_reid_output.mp4', help='Path to output video file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Re-ID similarity threshold')
    
    try:
        args = parser.parse_args()
        print(f"Arguments parsed successfully:")
        print(f"  Model: {args.model}")
        print(f"  Broadcast: {args.broadcast}")
        print(f"  Tactical: {args.tacticam}")
        print(f"  Output: {args.output}")
        print(f"  Confidence: {args.conf}")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        
    except SystemExit as e:
        print(f"Error parsing arguments: {e}")
        print("Usage: python main_script.py --model path/to/model.pt [other options]")
        return
    
    # Validate input files
    print("\n=== Validating input files ===")
    
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("Please provide a valid path to your YOLOv11 model file (.pt)")
        return
    else:
        print(f"✓ Model file found: {args.model}")
    
    if not Path(args.broadcast).exists():
        print(f"ERROR: Broadcast video not found: {args.broadcast}")
        print("Please provide a valid path to your broadcast video file")
        return
    else:
        print(f"✓ Broadcast video found: {args.broadcast}")
    
    if not Path(args.tacticam).exists():
        print(f"ERROR: Tactical camera video not found: {args.tacticam}")
        print("Please provide a valid path to your tactical camera video file")
        return
    else:
        print(f"✓ Tactical camera video found: {args.tacticam}")
    
    print("\n=== All files validated successfully ===")
    
    logger.info(f"Processing football game videos:")
    logger.info(f"  Broadcast camera: {args.broadcast}")
    logger.info(f"  Tactical camera: {args.tacticam}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Output: {args.output}")
    
    try:
        # Initialize system with football-specific settings
        print("\n=== Initializing Re-ID System ===")
        reid_system = PlayerReIDSystem(args.model, args.conf)
        reid_system.tracker.similarity_threshold = args.similarity_threshold
        
        # Process the football videos
        print("\n=== Starting video processing ===")
        reid_system.process_videos(args.broadcast, args.tacticam, args.output)
        print("\n=== Video processing completed ===")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()