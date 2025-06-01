#!/usr/bin/env python3
"""
Test Tesseract OCR configuration and capabilities
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
from config import Config

def test_tesseract_installation():
    """Test if Tesseract is properly installed and configured"""
    print("=== Testing Tesseract Installation ===")
    
    try:
        # Set Tesseract path if configured
        if hasattr(Config, 'TESSERACT_CMD') and Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
            print(f"Using Tesseract path: {Config.TESSERACT_CMD}")
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tesseract installation error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Tesseract is installed")
        print("2. Check the TESSERACT_CMD path in config.py")
        print("3. On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("4. On Linux: sudo apt-get install tesseract-ocr")
        print("5. On macOS: brew install tesseract")
        return False

def test_ocr_configurations():
    """Test different OCR configurations with sample text"""
    print("\n=== Testing OCR Configurations ===")
    
    # Create test images with different license plate formats
    test_cases = [
        "RJ14CV0002",  # Indian format
        "ABC123",      # Simple format
        "XYZ9876",     # Another simple format
        "MH12AB1234",  # Maharashtra format
        "DL01CA1234"   # Delhi format
    ]
    
    configs = [
        (r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "PSM 7 (Single text line)"),
        (r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "PSM 8 (Single word)"),
        (r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "PSM 6 (Single uniform block)"),
        (r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "PSM 13 (Raw line)")
    ]
    
    for test_text in test_cases:
        print(f"\n--- Testing with text: {test_text} ---")
        
        # Create a test image with the text
        img = create_test_plate_image(test_text)
        
        # Test each configuration
        for config, config_name in configs:
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(img)
                
                # Extract text
                result = pytesseract.image_to_string(pil_img, config=config).strip()
                
                # Clean result
                cleaned_result = ''.join(c for c in result.upper() if c.isalnum())
                
                # Check accuracy
                if cleaned_result == test_text:
                    print(f"  ✓ {config_name}: '{cleaned_result}' (PERFECT)")
                elif cleaned_result in test_text or test_text in cleaned_result:
                    print(f"  ~ {config_name}: '{cleaned_result}' (PARTIAL)")
                else:
                    print(f"  ✗ {config_name}: '{cleaned_result}' (FAILED)")
                    
            except Exception as e:
                print(f"  ✗ {config_name}: Error - {e}")

def create_test_plate_image(text):
    """Create a synthetic license plate image for testing"""
    # Create a white background
    img = np.ones((80, 300, 3), dtype=np.uint8) * 255
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Calculate text size and position
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (gray.shape[1] - text_width) // 2
    y = (gray.shape[0] + text_height) // 2
    
    # Add black text
    cv2.putText(gray, text, (x, y), font, font_scale, 0, thickness)
    
    # Add some noise and blur to simulate real conditions
    noise = np.random.normal(0, 10, gray.shape).astype(np.uint8)
    gray = cv2.add(gray, noise)
    
    # Slight blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray

def test_preprocessing_methods():
    """Test different image preprocessing methods"""
    print("\n=== Testing Preprocessing Methods ===")
    
    test_text = "RJ14CV0002"
    original_img = create_test_plate_image(test_text)
    
    # Add some distortions to make it more challenging
    # Add perspective distortion
    rows, cols = original_img.shape
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[10, 5], [cols-5, 10], [5, rows-10], [cols-10, rows-5]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted_img = cv2.warpPerspective(original_img, matrix, (cols, rows))
    
    # Add noise
    noise = np.random.normal(0, 20, distorted_img.shape).astype(np.uint8)
    noisy_img = cv2.add(distorted_img, noise)
    
    print(f"Testing with distorted version of: {test_text}")
    
    # Test different preprocessing methods
    methods = [
        ("Original", noisy_img),
        ("Gaussian Blur", cv2.GaussianBlur(noisy_img, (5, 5), 0)),
        ("Bilateral Filter", cv2.bilateralFilter(noisy_img, 11, 17, 17)),
        ("CLAHE", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(noisy_img)),
        ("Adaptive Threshold", cv2.adaptiveThreshold(noisy_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("Otsu Threshold", cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    ]
    
    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    for method_name, processed_img in methods:
        try:
            pil_img = Image.fromarray(processed_img)
            result = pytesseract.image_to_string(pil_img, config=config).strip()
            cleaned_result = ''.join(c for c in result.upper() if c.isalnum())
            
            if cleaned_result == test_text:
                print(f"  ✓ {method_name}: '{cleaned_result}' (PERFECT)")
            elif cleaned_result in test_text or test_text in cleaned_result:
                print(f"  ~ {method_name}: '{cleaned_result}' (PARTIAL)")
            else:
                print(f"  ✗ {method_name}: '{cleaned_result}' (FAILED)")
                
        except Exception as e:
            print(f"  ✗ {method_name}: Error - {e}")

def main():
    """Main test function"""
    print("=== Tesseract OCR Configuration Test ===\n")
    
    # Test installation
    if not test_tesseract_installation():
        return
    
    # Test configurations
    test_ocr_configurations()
    
    # Test preprocessing
    test_preprocessing_methods()
    
    print("\n=== Test Summary ===")
    print("If you see mostly PERFECT or PARTIAL results, your OCR setup is working well.")
    print("If you see many FAILED results, consider:")
    print("1. Adjusting image preprocessing parameters")
    print("2. Trying different PSM modes")
    print("3. Improving image quality (resolution, contrast, noise reduction)")
    print("4. Fine-tuning the character whitelist for your region")

if __name__ == "__main__":
    main()
