import easyocr
import cv2
import numpy as np
import re
import time
import threading
from collections import deque, Counter
from typing import List, Tuple, Optional, Dict
import queue

class RealTimeANPR:
    def __init__(self, languages=['en'], gpu=True, frame_skip=3, stability_frames=5):
        """
        Real-time ANPR system with EasyOCR for live video feeds
        
        Args:
            languages: Languages for OCR
            gpu: Use GPU acceleration
            frame_skip: Process every Nth frame (for performance)
            stability_frames: Number of consistent readings needed
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.stability_frames = stability_frames
        
        # For temporal consistency
        self.recent_results = deque(maxlen=stability_frames)
        self.last_stable_result = ""
        self.result_confidence = 0.0
        
        # Performance tracking
        self.processing_times = deque(maxlen=10)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Threading for non-blocking OCR
        self.ocr_queue = queue.Queue(maxsize=2)  # Limit queue size
        self.result_queue = queue.Queue()
        self.ocr_thread = None
        self.stop_threading = False
    
    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for real-time performance"""
        if plate_img is None or plate_img.size == 0:
            return None
            
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Quick size check and resize if needed
        height, width = gray.shape
        if height < 20 or width < 60:  # Too small for reliable OCR
            return None
            
        # Resize for optimal OCR (faster than complex preprocessing)
        if height < 32:
            scale_factor = 32 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_LINEAR)
        elif height > 64:  # Don't make it too large
            scale_factor = 64 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 64), interpolation=cv2.INTER_LINEAR)
        
        # Simple but effective preprocessing
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text_fast(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Fast text extraction optimized for real-time processing
        """
        start_time = time.time()
        
        # Preprocess
        processed_img = self.preprocess_plate_image(plate_img)
        if processed_img is None:
            return "", 0.0
        
        try:
            # OCR with optimized parameters for speed
            results = self.reader.readtext(
                processed_img,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                width_ths=0.8,  # More restrictive for speed
                height_ths=0.8,
                paragraph=False,
                detail=1,
                batch_size=1  # Process one at a time for real-time
            )
            
            # Process results quickly
            best_text = ""
            best_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Lower threshold for real-time
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if 4 <= len(cleaned_text) <= 10 and confidence > best_confidence:
                        best_text = cleaned_text
                        best_confidence = confidence
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return best_text, best_confidence
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0
    
    def get_stable_result(self, current_result: str, confidence: float) -> Tuple[str, float, bool]:
        """
        Get temporally stable result using recent history
        
        Returns:
            (stable_text, confidence, is_new_result)
        """
        if confidence > 0.6 and len(current_result) >= 4:
            self.recent_results.append(current_result)
        
        if len(self.recent_results) < 2:
            return self.last_stable_result, self.result_confidence, False
        
        # Find most common result in recent frames
        result_counts = Counter(self.recent_results)
        most_common = result_counts.most_common(1)[0]
        
        # If we have enough consistent readings
        if most_common[1] >= min(3, len(self.recent_results) // 2):
            new_result = most_common[0]
            if new_result != self.last_stable_result:
                self.last_stable_result = new_result
                self.result_confidence = confidence
                return new_result, confidence, True
        
        return self.last_stable_result, self.result_confidence, False
    
    def process_frame_threaded(self, frame: np.ndarray, plate_bbox: Tuple[int, int, int, int]):
        """
        Add frame to processing queue (non-blocking)
        """
        if not self.ocr_queue.full():
            try:
                x1, y1, x2, y2 = plate_bbox
                plate_crop = frame[y1:y2, x1:x2].copy()
                self.ocr_queue.put_nowait(plate_crop)
            except:
                pass
    
    def ocr_worker(self):
        """
        Background OCR processing thread
        """
        while not self.stop_threading:
            try:
                plate_img = self.ocr_queue.get(timeout=0.1)
                if plate_img is not None:
                    text, confidence = self.extract_text_fast(plate_img)
                    stable_text, stable_conf, is_new = self.get_stable_result(text, confidence)
                    
                    self.result_queue.put({
                        'text': stable_text,
                        'confidence': stable_conf,
                        'is_new': is_new,
                        'raw_text': text,
                        'raw_confidence': confidence
                    })
                
                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OCR Worker Error: {e}")
    
    def start_threading(self):
        """Start background OCR processing"""
        if self.ocr_thread is None or not self.ocr_thread.is_alive():
            self.stop_threading = False
            self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
            self.ocr_thread.start()
    
    def stop_background_processing(self):
        """Stop background OCR processing"""
        self.stop_threading = True
        if self.ocr_thread and self.ocr_thread.is_alive():
            self.ocr_thread.join(timeout=1.0)
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest OCR result (non-blocking)"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed"""
        self.frame_count += 1
        return self.frame_count % self.frame_skip == 0
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed > 1.0:  # Update every second
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
        else:
            fps = 0
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'fps': fps,
            'avg_processing_time': avg_processing_time,
            'queue_size': self.ocr_queue.qsize(),
            'stable_result': self.last_stable_result,
            'confidence': self.result_confidence
        }

def live_anpr_demo(camera_id=0, use_threading=True):
    """
    Demo function for real-time ANPR with live camera feed
    """
    # Initialize ANPR system
    anpr = RealTimeANPR(gpu=False, frame_skip=2, stability_frames=5)
    
    if use_threading:
        anpr.start_threading()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting live ANPR feed. Press 'q' to quit.")
    print("Click and drag to select plate region")
    
    # Variables for manual plate selection
    selecting = False
    selected_bbox = None
    start_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selected_bbox, start_point
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            if start_point:
                x1, y1 = start_point
                x2, y2 = x, y
                # Ensure proper bbox format
                selected_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    cv2.namedWindow('Live ANPR Feed', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Live ANPR Feed', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        anpr.fps_counter += 1
        display_frame = frame.copy()
        
        # Process frame if bbox is selected
        if selected_bbox and anpr.should_process_frame():
            x1, y1, x2, y2 = selected_bbox
            
            if use_threading:
                # Non-blocking processing
                anpr.process_frame_threaded(frame, selected_bbox)
                result = anpr.get_latest_result()
                
                if result and result['is_new']:
                    print(f"New stable result: {result['text']} (confidence: {result['confidence']:.2f})")
            else:
                # Blocking processing
                plate_crop = frame[y1:y2, x1:x2]
                text, confidence = anpr.extract_text_fast(plate_crop)
                stable_text, stable_conf, is_new = anpr.get_stable_result(text, confidence)
                
                if is_new:
                    print(f"Detected: {stable_text} (confidence: {stable_conf:.2f})")
        
        # Draw selection rectangle
        if selected_bbox:
            x1, y1, x2, y2 = selected_bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display current stable result
            if anpr.last_stable_result:
                cv2.putText(display_frame, f"Plate: {anpr.last_stable_result}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display performance stats
        stats = anpr.get_performance_stats()
        if stats['fps'] > 0:
            cv2.putText(display_frame, f"FPS: {stats['fps']:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if stats['avg_processing_time'] > 0:
            cv2.putText(display_frame, f"OCR Time: {stats['avg_processing_time']*1000:.1f}ms", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "Click and drag to select plate area", 
                   (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Live ANPR Feed', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    anpr.stop_background_processing()
    cap.release()
    cv2.destroyAllWindows()

# Example integration with automatic plate detection (YOLO, etc.)
def integrate_with_yolo_detection():
    """
    Example showing integration with automatic plate detection
    (Replace with your actual detection model)
    """
    anpr = RealTimeANPR(gpu=True, frame_skip=1)
    anpr.start_threading()
    
    cap = cv2.VideoCapture(0)
    
    # TODO: Load your plate detection model here
    # yolo_model = YOLO('path_to_your_plate_detection_model.pt')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # TODO: Replace with actual plate detection
        # detections = yolo_model(frame)
        # 
        # for detection in detections:
        #     if detection.confidence > 0.5:  # Plate detection confidence
        #         bbox = detection.bbox  # (x1, y1, x2, y2)
        #         anpr.process_frame_threaded(frame, bbox)
        
        # Get and display results
        result = anpr.get_latest_result()
        if result and result['is_new']:
            print(f"Auto-detected plate: {result['text']} (conf: {result['confidence']:.2f})")
        
        cv2.imshow('Auto ANPR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    anpr.stop_background_processing()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run live demo
    live_anpr_demo(camera_id=0, use_threading=True)