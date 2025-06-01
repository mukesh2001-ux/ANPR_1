from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class DetectedPlate(db.Model):
    """Model for storing detected number plates"""
    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String(20), nullable=True)  # OCR extracted text
    confidence = db.Column(db.Float, default=0.0)  # OCR confidence score
    image_path = db.Column(db.String(255), nullable=False)  # Path to saved image
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Detection metadata
    bbox_x = db.Column(db.Integer)  # Bounding box coordinates
    bbox_y = db.Column(db.Integer)
    bbox_width = db.Column(db.Integer)
    bbox_height = db.Column(db.Integer)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('detected_plates', lazy=True))
    
    def __repr__(self):
        return f'<DetectedPlate {self.plate_number} at {self.detected_at}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'plate_number': self.plate_number,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'detected_at': self.detected_at.isoformat(),
            'user_id': self.user_id,
            'bbox': {
                'x': self.bbox_x,
                'y': self.bbox_y,
                'width': self.bbox_width,
                'height': self.bbox_height
            }
        }

class SystemStats(db.Model):
    """Model for storing system statistics"""
    id = db.Column(db.Integer, primary_key=True)
    total_detections = db.Column(db.Integer, default=0)
    successful_ocr = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    @classmethod
    def get_stats(cls):
        """Get current system statistics"""
        stats = cls.query.first()
        if not stats:
            stats = cls()
            db.session.add(stats)
            db.session.commit()
        return stats
    
    def update_detection_count(self):
        """Increment detection count"""
        self.total_detections += 1
        self.last_updated = datetime.utcnow()
        db.session.commit()
    
    def update_ocr_success(self):
        """Increment successful OCR count"""
        self.successful_ocr += 1
        self.last_updated = datetime.utcnow()
        db.session.commit()
