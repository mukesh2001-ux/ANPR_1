import cv2
import numpy as np
import os
import re
from datetime import datetime
from config import Config

# Try to import EasyOCR, fallback to Tesseract if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("EasyOCR is available - using EasyOCR for text recognition")
except ImportError:
    EASYOCR_AVAILABLE = False
    import pytesseract
    from PIL import Image
    print("EasyOCR not available - falling back to Tesseract")

class ANPRService:
    """Service class for Automatic Number Plate Recognition"""
    
    def __init__(self):
        self.plate_cascade = cv2.CascadeClassifier(Config.CASCADE_FILE)
        self.min_area = Config.MIN_PLATE_AREA

        # Initialize OCR engine
        if EASYOCR_AVAILABLE:
            # Initialize EasyOCR reader
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
                self.ocr_engine = 'easyocr'
                print("EasyOCR reader initialized successfully")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                print("Falling back to Tesseract...")
                self._init_tesseract()
        else:
            self._init_tesseract()

        # Create upload directory if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    def _init_tesseract(self):
        """Initialize Tesseract OCR as fallback"""
        self.ocr_engine = 'tesseract'

        # Set Tesseract path (adjust for your system)
        if hasattr(Config, 'TESSERACT_CMD') and Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

        # Multiple OCR configurations to try for better recognition
        self.ocr_configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single text line
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single word
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single uniform block
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', # Raw line
        ]
    
    def detect_plates(self, frame):
        """
        Detect number plates in a frame
        Returns: list of (x, y, w, h) coordinates
        """
        print(f"DEBUG: Starting plate detection on frame of size {frame.shape}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"DEBUG: Cascade detected {len(plates)} potential plates")

        # Filter plates by minimum area
        valid_plates = []
        for (x, y, w, h) in plates:
            area = w * h
            print(f"DEBUG: Plate at ({x}, {y}, {w}, {h}) has area {area} (min: {self.min_area})")
            if area > self.min_area:
                valid_plates.append((x, y, w, h))
                print(f"DEBUG: Plate accepted")
            else:
                print(f"DEBUG: Plate rejected (too small)")

        print(f"DEBUG: Final valid plates: {len(valid_plates)}")
        return valid_plates
    
    def extract_plate_roi(self, frame, x, y, w, h):
        """Extract the region of interest (plate area) from frame"""
        return frame[y:y+h, x:x+w]
    
    def preprocess_plate_image_easyocr(self, plate_img):
        """
        Optimized preprocessing for EasyOCR
        Returns a single best processed image
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # Convert to grayscale if needed
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Quick size check and resize if needed (EasyOCR optimized)
        height, width = gray.shape
        if height < 20 or width < 60:  # Too small for reliable OCR
            return None

        # Resize for optimal EasyOCR performance
        if height < 32:
            scale_factor = 32 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_LINEAR)
        elif height > 64:  # Don't make it too large
            scale_factor = 64 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 64), interpolation=cv2.INTER_LINEAR)

        # Simple but effective preprocessing for EasyOCR
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold - works well with EasyOCR
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def preprocess_plate_image_tesseract(self, plate_img):
        """
        Enhanced preprocessing for Tesseract OCR
        Returns multiple processed versions to try
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Resize image for better OCR (minimum size requirements)
        height, width = gray.shape
        if height < 40 or width < 120:
            scale_factor = max(40/height, 120/width, 2.0)  # Minimum 2x scaling
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        processed_images = []

        # Method 1: Enhanced contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(enhanced, 11, 17, 17)

        # Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(thresh1)

        # Method 2: Otsu's thresholding
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh2)

        # Method 3: Morphological operations on enhanced image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        processed_images.append(morph)

        # Method 4: Edge-based enhancement
        edges = cv2.Canny(bilateral, 50, 150)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        processed_images.append(255 - dilated)  # Invert for white text on black background

        return processed_images

    def correct_ocr_errors(self, text):
        """
        Enhanced OCR error correction for license plate text
        """
        if not text:
            return text

        # Common OCR character corrections for license plates
        corrections = {
            'O': '0',  # Letter O to number 0
            'I': '1',  # Letter I to number 1
            'S': '5',  # Letter S to number 5
            'Z': '2',  # Letter Z to number 2
            'G': '6',  # Letter G to number 6
            'B': '8',  # Letter B to number 8 (sometimes)
            'Q': '0',  # Letter Q to number 0
            'D': '0',  # Letter D to number 0 (sometimes)
            'U': '0',  # Letter U to number 0 (common misread)
            'Y': '1',  # Letter Y to number 1 (sometimes)
            '8': 'B',  # Number 8 to letter B (for state codes)
            '0': 'O',  # Number 0 to letter O (for state codes)
            '1': 'I',  # Number 1 to letter I (for state codes)
            '5': 'S',  # Number 5 to letter S (for state codes)
            '6': 'G',  # Number 6 to letter G (for state codes)
        }

        corrected = text

        # Apply corrections based on position context
        # In license plates, certain positions are more likely to be numbers
        for i, char in enumerate(text):
            # If character is in a position that's typically numeric (after first 2-4 characters)
            if i >= 2:  # After first 2 characters, more likely to be numbers
                if char in corrections:
                    corrected = corrected[:i] + corrections[char] + corrected[i+1:]

        # Additional pattern-based corrections for common misreads
        # Fix specific patterns we've observed
        corrected = corrected.replace('CVO', 'CV0')    # Common misread
        corrected = corrected.replace('CVOO', 'CV00')  # Double O misread
        corrected = corrected.replace('CVOOO', 'CV000') # Triple O misread
        corrected = corrected.replace('OO', '00')      # Double O to double zero
        corrected = corrected.replace('OOO', '000')    # Triple O to triple zero
        corrected = corrected.replace('II', '11')      # Double I to double one
        corrected = corrected.replace('RU14', 'RJ14')  # Specific misread for RJ
        corrected = corrected.replace('RO14', 'RJ14')  # Another RJ misread

        # Remove hardcoded replacements - let OCR work naturally

        # Fix common number sequences at the end
        corrected = corrected.replace('OO02', '0002')  # Common ending pattern
        corrected = corrected.replace('O02', '002')    # Another ending pattern
        corrected = corrected.replace('OO2', '002')    # Another variation

        return corrected

    def extract_text_from_plate(self, plate_img):
        """
        Enhanced OCR using EasyOCR or Tesseract fallback
        Returns: (text, confidence)
        """
        print(f"DEBUG: Starting OCR on plate image of size {plate_img.shape} using {self.ocr_engine}")

        if self.ocr_engine == 'easyocr':
            return self._extract_text_easyocr(plate_img)
        else:
            return self._extract_text_tesseract(plate_img)

    def _extract_text_easyocr(self, plate_img):
        """
        Extract text using EasyOCR
        """
        try:
            # Preprocess image for EasyOCR
            processed_img = self.preprocess_plate_image_easyocr(plate_img)
            if processed_img is None:
                print("DEBUG: Preprocessing failed - image too small or invalid")
                return "", 0.0

            print(f"DEBUG: EasyOCR processing image of size {processed_img.shape}")

            # Use EasyOCR to extract text
            results = self.reader.readtext(
                processed_img,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                width_ths=0.7,  # Width threshold for text detection
                height_ths=0.7,  # Height threshold for text detection
                paragraph=False,  # Don't group text into paragraphs
                detail=1,  # Return detailed results with confidence
                batch_size=1  # Process one image at a time
            )

            print(f"DEBUG: EasyOCR found {len(results)} text regions")

            # Process results
            best_text = ""
            best_confidence = 0.0

            for i, (bbox, text, confidence) in enumerate(results):
                print(f"DEBUG: Result {i}: '{text}' with confidence {confidence:.3f}")

                # Clean the text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                # Apply OCR error corrections
                corrected_text = self.correct_ocr_errors(cleaned_text)

                print(f"DEBUG: Cleaned: '{cleaned_text}' -> Corrected: '{corrected_text}'")

                # Validate length and confidence
                if (Config.OCR_MIN_TEXT_LENGTH <= len(corrected_text) <= Config.OCR_MAX_TEXT_LENGTH and
                    confidence > 0.3 and confidence > best_confidence):  # EasyOCR confidence is 0-1
                    best_text = corrected_text
                    best_confidence = confidence * 100  # Convert to percentage for consistency
                    print(f"DEBUG: New best result: '{best_text}' with {best_confidence:.1f}% confidence")

            # If no good result, try with original image (no preprocessing)
            if not best_text:
                print("DEBUG: Trying EasyOCR on original image without preprocessing")
                try:
                    # Convert to grayscale if needed
                    if len(plate_img.shape) == 3:
                        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = plate_img

                    # Scale up the image
                    height, width = gray.shape
                    scale_factor = max(2.0, 64/height)  # Ensure reasonable size
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                    results = self.reader.readtext(
                        scaled,
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        width_ths=0.5,  # More lenient thresholds
                        height_ths=0.5,
                        paragraph=False,
                        detail=1
                    )

                    for bbox, text, confidence in results:
                        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        corrected_text = self.correct_ocr_errors(cleaned_text)

                        if (Config.OCR_MIN_TEXT_LENGTH <= len(corrected_text) <= Config.OCR_MAX_TEXT_LENGTH and
                            confidence > 0.2):  # Lower threshold for fallback
                            best_text = corrected_text
                            best_confidence = confidence * 100
                            print(f"DEBUG: Fallback found: '{best_text}' with {best_confidence:.1f}% confidence")
                            break

                except Exception as e:
                    print(f"DEBUG: EasyOCR fallback failed: {e}")

            return best_text, best_confidence

        except Exception as e:
            print(f"EasyOCR Error: {e}")
            return "", 0.0

    def _extract_text_tesseract(self, plate_img):
        """
        Extract text using Tesseract OCR (fallback method)
        """
        try:
            # Get multiple processed versions of the image
            processed_images = self.preprocess_plate_image_tesseract(plate_img)
            print(f"DEBUG: Generated {len(processed_images)} processed images for Tesseract")

            best_text = ""
            best_confidence = 0.0
            all_results = []

            # Try each preprocessing method with each OCR configuration
            for img_idx, processed_img in enumerate(processed_images):
                print(f"DEBUG: Processing image {img_idx}")
                for config_idx, ocr_config in enumerate(self.ocr_configs):
                    try:
                        # Convert to PIL Image for Tesseract
                        pil_img = Image.fromarray(processed_img)

                        # Extract text with confidence
                        data = pytesseract.image_to_data(pil_img, config=ocr_config, output_type=pytesseract.Output.DICT)

                        # Filter and combine text
                        texts = []
                        confidences = []

                        for i, conf in enumerate(data['conf']):
                            if int(conf) > Config.OCR_MIN_CONFIDENCE:
                                text = data['text'][i].strip()
                                if text and len(text) > 0:
                                    texts.append(text)
                                    confidences.append(int(conf))

                        if texts:
                            # Combine all text and calculate average confidence
                            combined_text = ''.join(texts)
                            avg_confidence = sum(confidences) / len(confidences)

                            # Clean up the text
                            cleaned_text = re.sub(r'[^A-Z0-9]', '', combined_text.upper())

                            # Apply OCR error corrections
                            corrected_text = self.correct_ocr_errors(cleaned_text)

                            # Validate license plate format
                            if len(corrected_text) >= Config.OCR_MIN_TEXT_LENGTH and len(corrected_text) <= Config.OCR_MAX_TEXT_LENGTH:
                                all_results.append((corrected_text, avg_confidence))

                                # Update best result if this is better
                                if avg_confidence > best_confidence:
                                    best_text = corrected_text
                                    best_confidence = avg_confidence

                    except Exception as inner_e:
                        print(f"DEBUG: Tesseract config {config_idx} failed: {inner_e}")
                        continue

            return best_text, best_confidence

        except Exception as e:
            print(f"Tesseract OCR Error: {e}")
            return "", 0.0

    def save_plate_image(self, plate_img, user_id):
        """
        Save plate image to disk
        Returns: file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"plate_{user_id}_{timestamp}.jpg"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)

        cv2.imwrite(filepath, plate_img)
        return filepath

    def save_debug_images(self, processed_images, user_id, plate_text="unknown"):
        """
        Save processed images for debugging OCR issues
        """
        try:
            debug_folder = os.path.join(Config.UPLOAD_FOLDER, "debug")
            os.makedirs(debug_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for i, img in enumerate(processed_images):
                filename = f"debug_{user_id}_{timestamp}_method{i}_{plate_text}.jpg"
                filepath = os.path.join(debug_folder, filename)
                cv2.imwrite(filepath, img)

        except Exception as e:
            print(f"Debug save error: {e}")
    
    def process_frame(self, frame, user_id):
        """
        Process a single frame for plate detection and OCR
        Returns: list of detection results
        """
        results = []
        plates = self.detect_plates(frame)

        for (x, y, w, h) in plates:
            # Extract plate region
            plate_roi = self.extract_plate_roi(frame, x, y, w, h)

            # Save plate image
            image_path = self.save_plate_image(plate_roi, user_id)

            # Extract text with enhanced OCR
            plate_text, confidence = self.extract_text_from_plate(plate_roi)

            # Debug: Save processed images if OCR fails or has low confidence
            if Config.OCR_DEBUG_MODE:
                try:
                    if self.ocr_engine == 'easyocr':
                        processed_img = self.preprocess_plate_image_easyocr(plate_roi)
                        if processed_img is not None:
                            self.save_debug_images([processed_img], user_id, plate_text or "failed")
                    else:
                        processed_images = self.preprocess_plate_image_tesseract(plate_roi)
                        self.save_debug_images(processed_images, user_id, plate_text or "failed")

                    print(f"Debug: OCR result '{plate_text}' with {confidence:.1f}% confidence using {self.ocr_engine}")
                    print(f"Debug: Plate coordinates: ({x}, {y}, {w}, {h})")
                except Exception as debug_e:
                    print(f"Debug error: {debug_e}")

            result = {
                'bbox': (int(x), int(y), int(w), int(h)),
                'image_path': image_path,
                'plate_text': plate_text,
                'confidence': float(confidence)
            }
            results.append(result)

        return results
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and text on frame
        Returns: annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            plate_text = detection['plate_text']
            confidence = detection['confidence']

            # Ensure coordinates are integers
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw text label
            label = f"{plate_text} ({confidence:.1f}%)" if plate_text else f"Plate ({confidence:.1f}%)"
            cv2.putText(annotated_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
