import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import threading
import queue
import time

# Import the enhanced runner script components
from enhanced_runner_script import ImprovedPlayerReIDSystem, DualCameraPlayerTracker

class DualCameraManager:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.reid_system = None
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            "model_path": "yolo11n.pt",
            "confidence_threshold": 0.5,
            "similarity_threshold": 0.7,
            "max_disappear_frames": 30,
            "id_stability_frames": 10,
            "output_settings": {
                "fps": 30,
                "codec": "mp4v",
                "quality": "high"
            },
            "display_settings": {
                "show_confidence": True,
                "show_camera_labels": True,
                "label_font_size": 0.8,
                "box_thickness": 3
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dual_camera_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Dual Camera Manager initialized")
    
    def initialize_system(self):
        """Initialize the ReID system with configuration"""
        try:
            self.reid_system = ImprovedPlayerReIDSystem(
                model_path=self.config["model_path"],
                conf_threshold=self.config["confidence_threshold"]
            )
            
            # Update tracker settings
            self.reid_system.tracker.similarity_threshold = self.config["similarity_threshold"]
            self.reid_system.tracker.max_disappear_frames = self.config["max_disappear_frames"]
            self.reid_system.tracker.id_stability_frames = self.config["id_stability_frames"]
            
            self.logger.info("ReID system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ReID system: {e}")
            return False
    
    def validate_inputs(self, video_path1, video_path2, output_path):
        """Validate input parameters"""
        errors = []
        
        if not os.path.exists(video_path1):
            errors.append(f"Video 1 not found: {video_path1}")
        
        if not os.path.exists(video_path2):
            errors.append(f"Video 2 not found: {video_path2}")
        
        # Check if output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")
        
        # Check file extensions
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for path in [video_path1, video_path2]:
            if not any(path.lower().endswith(ext) for ext in valid_extensions):
                errors.append(f"Unsupported video format: {path}")
        
        return errors
    
    def process_videos_async(self, video_path1, video_path2, output_path):
        """Process videos asynchronously with progress tracking"""
        def processing_thread():
            try:
                self.logger.info(f"Starting processing: {video_path1} + {video_path2}")
                
                # Add to processing queue
                self.processing_queue.put({
                    'status': 'started',
                    'timestamp': datetime.now(),
                    'inputs': [video_path1, video_path2],
                    'output': output_path
                })
                
                # Process videos
                self.reid_system.process_videos(video_path1, video_path2, output_path)
                
                # Add completion to results queue
                self.results_queue.put({
                    'status': 'completed',
                    'timestamp': datetime.now(),
                    'output': output_path,
                    'success': True
                })
                
            except Exception as e:
                self.logger.error(f"Processing failed: {e}")
                self.results_queue.put({
                    'status': 'failed',
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'success': False
                })
        
        # Start processing thread
        thread = threading.Thread(target=processing_thread)
        thread.daemon = True
        thread.start()
        
        return thread
    
    def get_processing_status(self):
        """Get current processing status"""
        status = {
            'queue_size': self.processing_queue.qsize(),
            'recent_results': []
        }
        
        # Get recent results
        while not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                status['recent_results'].append(result)
            except queue.Empty:
                break
        
        return status
    
    def create_batch_processor(self, batch_config):
        """Create a batch processor for multiple video pairs"""
        class BatchProcessor:
            def __init__(self, manager, config):
                self.manager = manager
                self.config = config
                self.results = []
            
            def process_batch(self, video_pairs):
                """Process multiple video pairs"""
                for i, (video1, video2, output) in enumerate(video_pairs):
                    self.manager.logger.info(f"Processing batch item {i+1}/{len(video_pairs)}")
                    
                    # Validate inputs
                    errors = self.manager.validate_inputs(video1, video2, output)
                    if errors:
                        self.results.append({
                            'index': i,
                            'success': False,
                            'errors': errors
                        })
                        continue
                    
                    # Process videos
                    try:
                        thread = self.manager.process_videos_async(video1, video2, output)
                        thread.join()  # Wait for completion
                        
                        self.results.append({
                            'index': i,
                            'success': True,
                            'output': output
                        })
                        
                    except Exception as e:
                        self.results.append({
                            'index': i,
                            'success': False,
                            'error': str(e)
                        })
                
                return self.results
        
        return BatchProcessor(self, batch_config)
    
    def export_tracking_data(self, output_path, tracking_data):
        """Export tracking data to JSON format"""
        try:
            export_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config,
                    'total_players': len(set(player_id for frame_data in tracking_data.values() 
                                           for player_id, _ in frame_data))
                },
                'tracking_data': tracking_data
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Tracking data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export tracking data: {e}")
            return False
    
    def generate_report(self, output_dir):
        """Generate a comprehensive processing report"""
        report_data = {
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'model_path': self.config["model_path"]
            },
            'processing_stats': self.get_processing_status(),
            'performance_metrics': {
                'total_players_tracked': len(self.reid_system.tracker.tracked_players) if self.reid_system else 0,
                'active_cross_camera_matches': len(self.reid_system.tracker.cross_camera_matches) if self.reid_system else 0
            }
        }
        
        # Create report directory
        report_dir = Path(output_dir)
        report_dir.mkdir(exist_ok=True)
        
        # Generate HTML report
        html_report = self.generate_html_report(report_data)
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = report_dir / f"report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # HTML report
        html_path = report_dir / f"report_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"Reports generated: {json_path}, {html_path}")
        return json_path, html_path
    
    def generate_html_report(self, report_data):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dual Camera Player Tracking Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
                .success { color: green; }
                .error { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dual Camera Player Tracking Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>System Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Model Path</td><td>{model_path}</td></tr>
                    <tr><td>Confidence Threshold</td><td>{confidence_threshold}</td></tr>
                    <tr><td>Similarity Threshold</td><td>{similarity_threshold}</td></tr>
                    <tr><td>Max Disappear Frames</td><td>{max_disappear_frames}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">
                    <strong>Total Players Tracked:</strong> {total_players}
                </div>
                <div class="metric">
                    <strong>Cross-Camera Matches:</strong> {cross_camera_matches}
                </div>
                <div class="metric">
                    <strong>Processing Queue Size:</strong> {queue_size}
                </div>
            </div>
            
            <div class="section">
                <h2>Recent Results</h2>
                <table>
                    <tr><th>Status</th><th>Timestamp</th><th>Details</th></tr>
                    {recent_results_rows}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Format recent results
        recent_results_rows = ""
        for result in report_data['processing_stats']['recent_results']:
            status_class = "success" if result.get('success', False) else "error"
            status_text = result.get('status', 'Unknown')
            timestamp = result.get('timestamp', 'N/A')
            details = result.get('output', result.get('error', 'N/A'))
            
            recent_results_rows += f"""
            <tr>
                <td class="{status_class}">{status_text}</td>
                <td>{timestamp}</td>
                <td>{details}</td>
            </tr>
            """
        
        return html_template.format(
            timestamp=report_data['system_info']['timestamp'],
            model_path=report_data['system_info']['model_path'],
            confidence_threshold=report_data['system_info']['config']['confidence_threshold'],
            similarity_threshold=report_data['system_info']['config']['similarity_threshold'],
            max_disappear_frames=report_data['system_info']['config']['max_disappear_frames'],
            total_players=report_data['performance_metrics']['total_players_tracked'],
            cross_camera_matches=report_data['performance_metrics']['active_cross_camera_matches'],
            queue_size=report_data['processing_stats']['queue_size'],
            recent_results_rows=recent_results_rows
        )

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Dual Camera Player Re-identification System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--video1', type=str, required=True, help='Path to first video')
    parser.add_argument('--video2', type=str, required=True, help='Path to second video')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--batch', type=str, help='Path to batch processing JSON file')
    parser.add_argument('--report', type=str, help='Generate report in specified directory')
    parser.add_argument('--export-tracking', type=str, help='Export tracking data to JSON file')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = DualCameraManager(args.config)
    
    if not manager.initialize_system():
        print("Failed to initialize system. Check logs for details.")
        return 1
    
    # Batch processing
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                batch_data = json.load(f)
            
            batch_processor = manager.create_batch_processor(batch_data.get('config', {}))
            results = batch_processor.process_batch(batch_data['video_pairs'])
            
            print(f"Batch processing completed. Results: {results}")
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            return 1
    
    # Single video processing
    else:
        # Validate inputs
        errors = manager.validate_inputs(args.video1, args.video2, args.output)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        
        # Process videos
        try:
            print("Starting video processing...")
            thread = manager.process_videos_async(args.video1, args.video2, args.output)
            
            # Monitor progress
            while thread.is_alive():
                time.sleep(1)
                status = manager.get_processing_status()
                if status['recent_results']:
                    for result in status['recent_results']:
                        print(f"Status: {result['status']} at {result['timestamp']}")
            
            print("Processing completed!")
            
        except Exception as e:
            print(f"Processing failed: {e}")
            return 1
    
    # Generate report
    if args.report:
        try:
            json_path, html_path = manager.generate_report(args.report)
            print(f"Report generated: {html_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")
    
    # Export tracking data
    if args.export_tracking:
        try:
            # This would need to be implemented to capture tracking data during processing
            print("Tracking data export functionality needs to be integrated with processing loop")
        except Exception as e:
            print(f"Tracking data export failed: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())