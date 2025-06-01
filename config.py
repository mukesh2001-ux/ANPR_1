import os
from datetime import timedelta

class Config:
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'anpr-secret-key-2024'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///anpr_system.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # ANPR Configuration
    FRAME_WIDTH = 1000
    FRAME_HEIGHT = 480
    MIN_PLATE_AREA = 500
    CASCADE_FILE = 'haarcascade_russian_plate_number.xml'

    # Auto-save configuration
    AUTO_SAVE_DETECTIONS = True
    MIN_CONFIDENCE_TO_SAVE = 5.0  # Minimum confidence to auto-save detection (lowered for testing)
    MAX_DETECTIONS_PER_MINUTE = 10  # Limit to prevent spam

    # Tesseract configuration (adjust path if needed)
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows default
    # For Linux/Mac: TESSERACT_CMD = '/usr/bin/tesseract'

    # OCR Enhancement Settings
    OCR_MIN_CONFIDENCE = 1   # Very low threshold to catch more text
    OCR_DEBUG_MODE = True    # Save debug images for failed OCR
    OCR_MIN_TEXT_LENGTH = 4  # Minimum text length for valid plates
    OCR_MAX_TEXT_LENGTH = 15 # Maximum text length for valid plates
    
    # SocketIO configuration
    SOCKETIO_ASYNC_MODE = 'threading'
    
    # Pagination
    PLATES_PER_PAGE = 20
