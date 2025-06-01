#!/usr/bin/env python3
"""
Test script to verify ANPR system setup
"""

import cv2
import pytesseract
import os
from config import Config

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access: OK")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera frame capture: OK ({frame.shape})")
            else:
                print("✗ Camera frame capture: Failed")
            cap.release()
        else:
            print("✗ Camera access: Failed")
            
        # Test Haar cascade
        if os.path.exists(Config.CASCADE_FILE):
            cascade = cv2.CascadeClassifier(Config.CASCADE_FILE)
            if not cascade.empty():
                print("✓ Haar cascade loading: OK")
            else:
                print("✗ Haar cascade loading: Failed")
        else:
            print(f"✗ Haar cascade file not found: {Config.CASCADE_FILE}")
            
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")

def test_tesseract():
    """Test Tesseract OCR installation"""
    print("\nTesting Tesseract OCR...")
    try:
        # Set Tesseract path if configured
        if hasattr(Config, 'TESSERACT_CMD') and Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Test OCR with simple text
        import numpy as np
        from PIL import Image
        
        # Create a simple test image with text
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "TEST123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(test_img)
        
        # Test OCR
        text = pytesseract.image_to_string(pil_img, config='--psm 8')
        if "TEST123" in text.upper():
            print("✓ OCR text recognition: OK")
        else:
            print(f"✗ OCR text recognition: Failed (got: '{text.strip()}')")
            
    except Exception as e:
        print(f"✗ Tesseract test failed: {e}")
        print("  Make sure Tesseract is installed and the path in config.py is correct")

def test_directories():
    """Test required directories"""
    print("\nTesting directories...")
    
    directories = [
        'static/uploads',
        'templates',
        'static/css',
        'static/js'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  Created directory: {directory}")
            except Exception as e:
                print(f"  Failed to create directory: {e}")

def test_database():
    """Test database connection"""
    print("\nTesting database...")
    try:
        from app import app, db
        with app.app_context():
            # Test database connection
            db.create_all()
            print("✓ Database connection: OK")
            print("✓ Database tables created: OK")
    except Exception as e:
        print(f"✗ Database test failed: {e}")

def main():
    """Run all tests"""
    print("ANPR System Setup Test")
    print("=" * 50)
    
    test_opencv()
    test_tesseract()
    test_directories()
    test_database()
    
    print("\n" + "=" * 50)
    print("Setup test completed!")
    print("\nIf all tests passed, you can run the application with:")
    print("python app.py")
    print("\nDefault login: admin / admin123")

if __name__ == "__main__":
    main()
