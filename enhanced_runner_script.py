import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import defaultdict
import math
import threading
import time
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

class GlobalPlayerTracker:
    def __init__(self, similarity_threshold=0.6, max_disappear_frames=30, id_stability_frames=5):
        self.similarity_threshold = similarity_threshold
        self.max_disappear_frames = max_disappear_frames
        self.id_stability_frames = id_stability_frames
        
        # Global player management
        self.next_global_id = 1
        self.global_players = {}  # global_id -> player_info
        self.camera_detections = {0: [], 1: []}  # camera_id -> [(global_id, detection)]
        
        # Player information storage
        self.player_features = {}  # global_id -> features
        self.player_positions = {}  # global_id -> {camera_id: (x, y)}
        self.player_last_seen = {}  # global_id -> frames_since_last_seen
        self.player_stability_count = {}  # global_id -> stability_count
        self.player_camera_history = {}  # global_id -> {camera_id: last_seen_frame}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Enhanced color palette
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
    
    def extract_enhanced_features(self, frame, bbox, camera_id=0):
        """Extract enhanced features from player bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)  # Extended feature vector
        
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(128)
        
        try:
            # Resize to standard size
            player_region = cv2.resize(player_region, (64, 128))
            
            # 1. Color histogram features (more detailed)
            hist_features = []
            for channel in range(3):
                hist = cv2.calcHist([player_region], [channel], None, [16], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-6)
                hist_features.extend(hist)
            
            # 2. Texture features using Local Binary Patterns
            gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            
            # Simple texture approximation
            edges = cv2.Canny(gray, 50, 150)
            texture_feature = np.mean(edges) / 255.0
            
            # 3. Body part features (upper/lower body color difference)
            upper_region = player_region[:int(player_region.shape[0]*0.6), :]
            lower_region = player_region[int(player_region.shape[0]*0.6):, :]
            
            upper_color = np.mean(upper_region, axis=(0, 1))
            lower_color = np.mean(lower_region, axis=(0, 1))
            
            # 4. Geometric features
            center_x = (x1 + x2) / (2 * w)
            center_y = (y1 + y2) / (2 * h)
            width_ratio = (x2 - x1) / w
            height_ratio = (y2 - y1) / h
            aspect_ratio = (x2 - x1) / (y2 - y1)
            
            # 5. Camera-specific normalization
            camera_norm = np.array([
                camera_id * 0.05,  # Slight camera bias
                np.sin(camera_id * np.pi/2) * 0.05,  # Viewing angle compensation
                np.cos(camera_id * np.pi/2) * 0.05
            ])
            
            # Combine all features
            geometric_features = np.array([
                center_x, center_y, width_ratio, height_ratio, aspect_ratio, texture_feature
            ])
            
            body_features = np.concatenate([upper_color, lower_color]) / 255.0
            
            # Construct final feature vector
            features = np.concatenate([
                hist_features,  # 48 features (16*3)
                geometric_features,  # 6 features
                body_features,  # 6 features
                camera_norm  # 3 features
            ])
            
            # Pad or truncate to exactly 128 features
            if len(features) > 128:
                features = features[:128]
            else:
                padding = np.zeros(128 - len(features))
                features = np.concatenate([features, padding])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(128)
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate enhanced similarity between feature vectors"""
        if np.linalg.norm(features1) == 0 or np.linalg.norm(features2) == 0:
            return 0.0
        
        # Cosine similarity
        cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        # Histogram intersection (for color features)
        hist1 = features1[:48]  # First 48 features are color histograms
        hist2 = features2[:48]
        hist_intersection = np.sum(np.minimum(hist1, hist2))
        
        # Combine similarities
        combined_similarity = 0.7 * cosine_sim + 0.3 * hist_intersection
        
        return max(0, combined_similarity)
    
    def cross_camera_assignment(self, detections_cam1, detections_cam2, frame1, frame2):
        """Assign global IDs to detections from both cameras using Hungarian algorithm"""
        with self.lock:
            # Extract features for all detections
            features_cam1 = []
            features_cam2 = []
            
            for det in detections_cam1:
                features = self.extract_enhanced_features(frame1, det[:4], camera_id=0)
                features_cam1.append(features)
            
            for det in detections_cam2:
                features = self.extract_enhanced_features(frame2, det[:4], camera_id=1)
                features_cam2.append(features)
            
            # Current frame assignments
            current_assignments = {0: [], 1: []}
            
            # Match with existing global players
            all_detections = [(det, feat, 0) for det, feat in zip(detections_cam1, features_cam1)] + \
                           [(det, feat, 1) for det, feat in zip(detections_cam2, features_cam2)]
            
            # Create cost matrix for assignment
            existing_players = list(self.player_features.keys())
            cost_matrix = []
            
            for detection, features, camera_id in all_detections:
                costs = []
                for global_id in existing_players:
                    if global_id in self.player_features:
                        similarity = self.calculate_feature_similarity(features, self.player_features[global_id])
                        
                        # Position continuity bonus
                        position_bonus = 0
                        if global_id in self.player_positions and camera_id in self.player_positions[global_id]:
                            old_pos = self.player_positions[global_id][camera_id]
                            new_pos = ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
                            
                            # Calculate normalized distance
                            if camera_id == 0:
                                max_dist = max(frame1.shape[1], frame1.shape[0]) * 0.3
                            else:
                                max_dist = max(frame2.shape[1], frame2.shape[0]) * 0.3
                            
                            distance = math.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
                            position_bonus = max(0, 1 - distance / max_dist) * 0.2
                        
                        # Camera history bonus
                        history_bonus = 0
                        if global_id in self.player_camera_history:
                            if camera_id in self.player_camera_history[global_id]:
                                history_bonus = 0.1
                        
                        total_similarity = similarity + position_bonus + history_bonus
                        costs.append(1 - total_similarity)  # Convert to cost
                    else:
                        costs.append(1.0)  # High cost for invalid players
                
                cost_matrix.append(costs)
            
            # Apply Hungarian algorithm for optimal assignment
            if cost_matrix and existing_players:
                cost_matrix = np.array(cost_matrix)
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Process assignments
                assigned_players = set()
                for row_idx, col_idx in zip(row_indices, col_indices):
                    if row_idx < len(all_detections) and col_idx < len(existing_players):
                        cost = cost_matrix[row_idx, col_idx]
                        if cost < (1 - self.similarity_threshold):  # Accept if similarity > threshold
                            detection, features, camera_id = all_detections[row_idx]
                            global_id = existing_players[col_idx]
                            
                            # Update player information
                            self.player_features[global_id] = features
                            if global_id not in self.player_positions:
                                self.player_positions[global_id] = {}
                            self.player_positions[global_id][camera_id] = (
                                (detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2
                            )
                            self.player_last_seen[global_id] = 0
                            self.player_stability_count[global_id] = min(
                                self.player_stability_count.get(global_id, 0) + 1,
                                self.id_stability_frames
                            )
                            
                            # Update camera history
                            if global_id not in self.player_camera_history:
                                self.player_camera_history[global_id] = {}
                            self.player_camera_history[global_id][camera_id] = 0
                            
                            current_assignments[camera_id].append((global_id, detection))
                            assigned_players.add(row_idx)
                
                # Create new players for unassigned detections
                for row_idx, (detection, features, camera_id) in enumerate(all_detections):
                    if row_idx not in assigned_players:
                        # Create new global player
                        global_id = self.next_global_id
                        self.next_global_id += 1
                        
                        self.player_features[global_id] = features
                        self.player_positions[global_id] = {
                            camera_id: ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
                        }
                        self.player_last_seen[global_id] = 0
                        self.player_stability_count[global_id] = 1
                        self.player_camera_history[global_id] = {camera_id: 0}
                        
                        current_assignments[camera_id].append((global_id, detection))
            
            else:
                # No existing players, create new ones
                for detection, features, camera_id in all_detections:
                    global_id = self.next_global_id
                    self.next_global_id += 1
                    
                    self.player_features[global_id] = features
                    self.player_positions[global_id] = {
                        camera_id: ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)
                    }
                    self.player_last_seen[global_id] = 0
                    self.player_stability_count[global_id] = 1
                    self.player_camera_history[global_id] = {camera_id: 0}
                    
                    current_assignments[camera_id].append((global_id, detection))
            
            # Update last seen for all players
            players_to_remove = []
            for global_id in list(self.player_last_seen.keys()):
                found_in_current = False
                for camera_assignments in current_assignments.values():
                    if any(pid == global_id for pid, _ in camera_assignments):
                        found_in_current = True
                        break
                
                if not found_in_current:
                    self.player_last_seen[global_id] += 1
                    if self.player_last_seen[global_id] > self.max_disappear_frames:
                        players_to_remove.append(global_id)
            
            # Remove lost players
            for global_id in players_to_remove:
                self.remove_player(global_id)
            
            # Update camera history for all players
            for global_id in self.player_camera_history:
                for cam_id in self.player_camera_history[global_id]:
                    self.player_camera_history[global_id][cam_id] += 1
            
            # Store current frame assignments
            self.camera_detections[0] = current_assignments[0]
            self.camera_detections[1] = current_assignments[1]
            
            # Return only stable players
            stable_assignments = {0: [], 1: []}
            for camera_id in [0, 1]:
                for global_id, detection in current_assignments[camera_id]:
                    if self.player_stability_count.get(global_id, 0) >= min(2, self.id_stability_frames):
                        stable_assignments[camera_id].append((global_id, detection))
            
            return stable_assignments[0], stable_assignments[1]
    
    def remove_player(self, global_id):
        """Remove a player from all tracking structures"""
        for attr in ['player_features', 'player_positions', 'player_last_seen', 
                     'player_stability_count', 'player_camera_history']:
            if hasattr(self, attr):
                attr_dict = getattr(self, attr)
                if global_id in attr_dict:
                    del attr_dict[global_id]
    
    def get_color(self, player_id):
        """Get consistent color for a player ID"""
        color_index = (player_id - 1) % len(self.colors)
        return self.colors[color_index]


class ImprovedPlayerReIDSystem:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.global_tracker = GlobalPlayerTracker()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_players(self, frame):
        """Detect players in frame"""
        results = self.model(frame, conf=self.conf_threshold, classes=[0])
        
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
    
    def draw_tracking_results(self, frame, tracked_players, camera_id=0):
        """Draw enhanced tracking results"""
        annotated_frame = frame.copy()
        
        for player_id, detection in tracked_players:
            x1, y1, x2, y2 = map(int, detection[:4])
            confidence = detection[4]
            
            # Get consistent color
            color = self.global_tracker.get_color(player_id)
            
            # Draw bounding box with enhanced styling
            thickness = 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label
            label = f"P{player_id}"
            if camera_id == 1:
                label += " (C2)"
            
            # Draw label with improved styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Background for label
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
            # Draw semi-transparent background
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, 
                         (x1, label_y - text_height - 5), 
                         (x1 + text_width + 10, label_y + 5), 
                         color, -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 5, label_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame
    
    def process_videos(self, video_path1, video_path2, output_path):
        """Process two videos with global cross-camera re-identification"""
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
        
        output_fps = min(fps1, fps2)
        output_width = max(width1, width2)
        output_height = height1 + height2
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
        
        frame_count = 0
        self.logger.info("Starting enhanced dual-camera processing with global re-identification...")
        
        try:
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                # Detect players in both frames
                detections1 = self.detect_players(frame1)
                detections2 = self.detect_players(frame2)
                
                # Resize frames for consistent output
                frame1_resized = cv2.resize(frame1, (output_width, height1))
                frame2_resized = cv2.resize(frame2, (output_width, height2))
                
                # Scale detections accordingly
                scale_x1 = output_width / width1
                scale_x2 = output_width / width2
                
                scaled_detections1 = []
                for det in detections1:
                    x1, y1, x2, y2, conf = det
                    scaled_detections1.append([x1 * scale_x1, y1, x2 * scale_x1, y2, conf])
                
                scaled_detections2 = []
                for det in detections2:
                    x1, y1, x2, y2, conf = det
                    scaled_detections2.append([x1 * scale_x2, y1, x2 * scale_x2, y2, conf])
                
                # Global cross-camera assignment
                tracked_players1, tracked_players2 = self.global_tracker.cross_camera_assignment(
                    scaled_detections1, scaled_detections2, frame1_resized, frame2_resized
                )
                
                # Draw results
                result_frame1 = self.draw_tracking_results(frame1_resized, tracked_players1, camera_id=0)
                result_frame2 = self.draw_tracking_results(frame2_resized, tracked_players2, camera_id=1)
                
                # Combine frames
                combined_frame = np.vstack([result_frame1, result_frame2])
                
                # Add separator line
                cv2.line(combined_frame, (0, height1), (output_width, height1), (255, 255, 255), 2)
                
                # Add camera labels
                cv2.putText(combined_frame, "Camera 1", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_frame, "Camera 2", (10, height1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add tracking info
                total_players = len(self.global_tracker.player_features)
                cv2.putText(combined_frame, f"Global Players: {total_players}", (10, output_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(combined_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    self.logger.info(f"Processed {frame_count} frames - Active players: {total_players}")
                
                # Display (optional)
                display_frame = cv2.resize(combined_frame, (960, 540))
                cv2.imshow('Enhanced Global Re-ID System', display_frame)
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
    reid_system = ImprovedPlayerReIDSystem('yolo11n.pt', conf_threshold=0.5)
    reid_system.process_videos('broadcast.mp4', 'tacticam.mp4', 'global_reid_output.mp4')