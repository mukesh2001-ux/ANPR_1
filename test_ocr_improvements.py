#!/usr/bin/env python3
"""
Test script to verify OCR improvements for license plate recognition
"""

import cv2
import os
import sys
from anpr_service import ANPRService
from config import Config

def test_ocr_with_sample_images():
    """Test OCR with sample images from the uploads folder"""
    print("Testing OCR improvements...")
    
    # Initialize ANPR service
    anpr = ANPRService()
    
    # Look for sample images in uploads folder
    uploads_folder = Config.UPLOAD_FOLDER
    if not os.path.exists(uploads_folder):
        print(f"Uploads folder not found: {uploads_folder}")
        return
    
    # Find recent plate images
    image_files = []
    for file in os.listdir(uploads_folder):
        if file.startswith('plate_') and file.endswith('.jpg'):
            image_files.append(file)
    
    if not image_files:
        print("No plate images found in uploads folder")
        print("Please run the ANPR system first to capture some plate images")
        return
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(uploads_folder, x)), reverse=True)
    
    print(f"Found {len(image_files)} plate images. Testing the 5 most recent...")
    
    # Test the 5 most recent images
    for i, filename in enumerate(image_files[:5]):
        filepath = os.path.join(uploads_folder, filename)
        print(f"\n--- Testing image {i+1}: {filename} ---")
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            print(f"Could not load image: {filepath}")
            continue
        
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Test OCR
        plate_text, confidence = anpr.extract_text_from_plate(img)
        
        print(f"OCR Result: '{plate_text}'")
        print(f"Confidence: {confidence:.1f}%")
        
        if plate_text:
            print(f"✓ SUCCESS: Detected text '{plate_text}' with {confidence:.1f}% confidence")
        else:
            print("✗ FAILED: No text detected")
            
            # Show preprocessing results for failed cases
            processed_images = anpr.preprocess_plate_image(img)
            print(f"Generated {len(processed_images)} processed versions for analysis")

def test_ocr_with_camera():
    """Test OCR with live camera feed"""
    print("\nTesting with live camera...")
    print("Press 'c' to capture and test OCR, 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    # Initialize ANPR service
    anpr = ANPRService()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect plates
        plates = anpr.detect_plates(frame)
        
        # Draw detections
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to test OCR", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('ANPR Test - Press c to capture, q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and len(plates) > 0:
            # Test OCR on first detected plate
            x, y, w, h = plates[0]
            plate_roi = anpr.extract_plate_roi(frame, x, y, w, h)
            
            plate_text, confidence = anpr.extract_text_from_plate(plate_roi)
            
            print(f"\nCaptured plate OCR result:")
            print(f"Text: '{plate_text}'")
            print(f"Confidence: {confidence:.1f}%")
            
            # Save the captured plate for analysis
            cv2.imwrite('test_capture.jpg', plate_roi)
            print("Saved captured plate as 'test_capture.jpg'")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main test function"""
    print("=== ANPR OCR Improvement Test ===")
    
    # Test with existing images first
    test_ocr_with_sample_images()
    
    # Ask user if they want to test with camera
    response = input("\nDo you want to test with live camera? (y/n): ").lower()
    if response == 'y':
        test_ocr_with_camera()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
