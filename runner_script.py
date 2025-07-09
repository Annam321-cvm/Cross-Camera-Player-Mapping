import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import defaultdict
import math

class ImprovedPlayerTracker:
    def __init__(self, similarity_threshold=0.7, max_disappear_frames=30, id_stability_frames=10):
        self.similarity_threshold = similarity_threshold
        self.max_disappear_frames = max_disappear_frames
        self.id_stability_frames = id_stability_frames
        
        self.next_id = 1
        self.tracked_players = {}
        self.player_features = {}
        self.player_positions = {}
        self.player_last_seen = {}
        self.player_stability_count = {}
        
        # Predefined colors for better visualization (max 20 players)
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 128, 0),  # Olive
            (0, 0, 128),    # Navy
            (128, 0, 0),    # Maroon
            (255, 192, 203), # Pink
            (165, 42, 42),  # Brown
            (128, 128, 128), # Gray
            (255, 20, 147), # Deep Pink
            (0, 191, 255),  # Deep Sky Blue
            (50, 205, 50),  # Lime Green
            (220, 20, 60),  # Crimson
            (255, 140, 0),  # Dark Orange
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, frame, bbox):
        """Extract features from player bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(64)
        
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(64)
        
        try:
            # Resize to standard size
            player_region = cv2.resize(player_region, (64, 128))
            
            # Extract color histogram features
            hist_b = cv2.calcHist([player_region], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([player_region], [1], None, [8], [0, 256])
            hist_r = cv2.calcHist([player_region], [2], None, [8], [0, 256])
            
            # Normalize histograms
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)
            
            # Combine features
            features = np.concatenate([hist_b, hist_g, hist_r])
            
            # Add position and size features (normalized)
            center_x = (x1 + x2) / (2 * w)
            center_y = (y1 + y2) / (2 * h)
            width_ratio = (x2 - x1) / w
            height_ratio = (y2 - y1) / h
            
            position_features = np.array([center_x, center_y, width_ratio, height_ratio])
            
            # Pad to reach 64 features
            remaining = 64 - len(features) - len(position_features)
            if remaining > 0:
                padding = np.zeros(remaining)
                features = np.concatenate([features, position_features, padding])
            else:
                features = np.concatenate([features[:60], position_features])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(64)
    
    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        if np.linalg.norm(features1) == 0 or np.linalg.norm(features2) == 0:
            return 0.0
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update(self, frame, detections):
        """Update tracker with new detections"""
        frame_height, frame_width = frame.shape[:2]
        current_frame_players = []
        
        # Extract features for all detections
        detection_features = []
        detection_positions = []
        
        for detection in detections:
            bbox = detection[:4]
            features = self.extract_features(frame, bbox)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            detection_features.append(features)
            detection_positions.append((center_x, center_y))
        
        # Match detections to existing players
        matched_players = set()
        
        for i, (features, position) in enumerate(zip(detection_features, detection_positions)):
            best_match_id = None
            best_similarity = 0
            
            for player_id, stored_features in self.player_features.items():
                if player_id in matched_players:
                    continue
                
                # Calculate feature similarity
                feature_similarity = self.calculate_similarity(features, stored_features)
                
                # Calculate position distance (normalized)
                if player_id in self.player_positions:
                    old_pos = self.player_positions[player_id]
                    position_distance = self.calculate_distance(position, old_pos)
                    max_distance = min(frame_width, frame_height) * 0.3  # Allow 30% of frame size movement
                    position_similarity = max(0, 1 - (position_distance / max_distance))
                else:
                    position_similarity = 0.5
                
                # Combined similarity (weighted)
                combined_similarity = 0.7 * feature_similarity + 0.3 * position_similarity
                
                if combined_similarity > best_similarity and combined_similarity > self.similarity_threshold:
                    best_similarity = combined_similarity
                    best_match_id = player_id
            
            if best_match_id is not None:
                # Update existing player
                self.player_features[best_match_id] = features
                self.player_positions[best_match_id] = position
                self.player_last_seen[best_match_id] = 0
                self.player_stability_count[best_match_id] = min(
                    self.player_stability_count.get(best_match_id, 0) + 1,
                    self.id_stability_frames
                )
                matched_players.add(best_match_id)
                current_frame_players.append((best_match_id, detections[i]))
            else:
                # Create new player
                new_id = self.next_id
                self.next_id += 1
                self.player_features[new_id] = features
                self.player_positions[new_id] = position
                self.player_last_seen[new_id] = 0
                self.player_stability_count[new_id] = 1
                matched_players.add(new_id)
                current_frame_players.append((new_id, detections[i]))
        
        # Update last seen for all players
        players_to_remove = []
        for player_id in self.player_last_seen:
            if player_id not in matched_players:
                self.player_last_seen[player_id] += 1
                if self.player_last_seen[player_id] > self.max_disappear_frames:
                    players_to_remove.append(player_id)
        
        # Remove lost players
        for player_id in players_to_remove:
            del self.player_features[player_id]
            del self.player_positions[player_id]
            del self.player_last_seen[player_id]
            if player_id in self.player_stability_count:
                del self.player_stability_count[player_id]
        
        # Only return stable players
        stable_players = []
        for player_id, detection in current_frame_players:
            if self.player_stability_count.get(player_id, 0) >= min(3, self.id_stability_frames):
                stable_players.append((player_id, detection))
        
        return stable_players
    
    def get_color(self, player_id):
        """Get consistent color for a player ID"""
        color_index = (player_id - 1) % len(self.colors)
        return self.colors[color_index]

class ImprovedPlayerReIDSystem:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = ImprovedPlayerTracker()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_players(self, frame):
        """Detect players in frame"""
        results = self.model(frame, conf=self.conf_threshold, classes=[0])  # 0 is person class
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Filter by confidence and reasonable size
                width = x2 - x1
                height = y2 - y1
                
                if conf > self.conf_threshold and width > 20 and height > 40:
                    detections.append([x1, y1, x2, y2, conf])
        
        return detections
    
    def draw_tracking_results(self, frame, tracked_players):
        """Draw clean tracking results"""
        annotated_frame = frame.copy()
        
        for player_id, detection in tracked_players:
            x1, y1, x2, y2 = map(int, detection[:4])
            confidence = detection[4]
            
            # Get consistent color
            color = self.tracker.get_color(player_id)
            
            # Draw bounding box with thicker, cleaner lines
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create clean label background
            label = f"Player {player_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(annotated_frame, 
                         (x1, label_y - text_height - 5), 
                         (x1 + text_width + 10, label_y + 5), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 5, label_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame
    
    def process_videos(self, video_path1, video_path2, output_path):
        """Process two videos with improved re-identification"""
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        
        if not cap1.isOpened() or not cap2.isOpened():
            self.logger.error("Could not open video files")
            return
        
        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use the minimum FPS for output
        output_fps = min(fps1, fps2)
        
        # Create output video writer
        output_width = max(width1, width2)
        output_height = height1 + height2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
        
        frame_count = 0
        self.logger.info("Starting video processing...")
        
        try:
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                # Detect players in both frames
                detections1 = self.detect_players(frame1)
                detections2 = self.detect_players(frame2)
                
                # Combine detections for tracking
                all_detections = detections1 + detections2
                
                # Create combined frame for feature extraction
                combined_frame = np.vstack([
                    cv2.resize(frame1, (output_width, height1)),
                    cv2.resize(frame2, (output_width, height2))
                ])
                
                # Adjust detection coordinates for frame2
                adjusted_detections2 = []
                for det in detections2:
                    x1, y1, x2, y2, conf = det
                    # Scale to match output width and offset by frame1 height
                    scale_x = output_width / width2
                    x1_adj = x1 * scale_x
                    x2_adj = x2 * scale_x
                    y1_adj = y1 + height1
                    y2_adj = y2 + height1
                    adjusted_detections2.append([x1_adj, y1_adj, x2_adj, y2_adj, conf])
                
                # Adjust detections1 for output width
                adjusted_detections1 = []
                for det in detections1:
                    x1, y1, x2, y2, conf = det
                    scale_x = output_width / width1
                    x1_adj = x1 * scale_x
                    x2_adj = x2 * scale_x
                    adjusted_detections1.append([x1_adj, y1, x2_adj, y2, conf])
                
                # Combine adjusted detections
                combined_detections = adjusted_detections1 + adjusted_detections2
                
                # Update tracker
                tracked_players = self.tracker.update(combined_frame, combined_detections)
                
                # Draw results
                result_frame = self.draw_tracking_results(combined_frame, tracked_players)
                
                # Write frame
                out.write(result_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    self.logger.info(f"Processed {frame_count} frames")
                
                # Display progress (optional)
                cv2.imshow('Football Player Re-ID', cv2.resize(result_frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
        finally:
            cap1.release()
            cap2.release()
            out.release()
            cv2.destroyAllWindows()
            
            self.logger.info(f"Processing complete. Output saved to {output_path}")
            self.logger.info(f"Total frames processed: {frame_count}")

# Usage example
if __name__ == "__main__":
    # Initialize system
    reid_system = ImprovedPlayerReIDSystem('yolo11n.pt', conf_threshold=0.5)
    
    # Process videos
    reid_system.process_videos('broadcast.mp4', 'tacticam.mp4', 'improved_output.mp4')