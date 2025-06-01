from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_socketio import SocketIO, emit
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import cv2
import base64
import threading
import time
import os
import csv
import io
from sqlalchemy import and_, or_, desc, asc

from config import Config
from models import db, User, DetectedPlate, SystemStats
from anpr_service import ANPRService

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
csrf = CSRFProtect(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=Config.SOCKETIO_ASYNC_MODE)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize ANPR Service
anpr_service = ANPRService()

# Global variables for camera
camera = None
detection_active = False
detection_thread = None

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', 
                             validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')

# Routes
@app.route('/favicon.ico')
def favicon():
    return send_file('static/favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if username or email already exists
        existing_user = User.query.filter(
            or_(User.username == form.username.data, User.email == form.email.data)
        ).first()
        
        if existing_user:
            flash('Username or email already exists', 'error')
        else:
            user = User(
                username=form.username.data,
                email=form.email.data
            )
            user.set_password(form.password.data)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get system statistics
    stats = SystemStats.get_stats()
    
    # Get user-specific statistics
    user_detections = DetectedPlate.query.filter_by(user_id=current_user.id).count()
    
    # Calculate accuracy
    total_with_text = DetectedPlate.query.filter(DetectedPlate.plate_number.isnot(None)).count()
    accuracy = (total_with_text / stats.total_detections * 100) if stats.total_detections > 0 else 0
    
    # Get recent detections
    recent_plates = DetectedPlate.query.filter_by(user_id=current_user.id)\
                                     .order_by(desc(DetectedPlate.detected_at))\
                                     .limit(5).all()
    
    return render_template('dashboard.html',
                         stats=stats,
                         user_detections=user_detections,
                         accuracy=round(accuracy, 1),
                         recent_plates=recent_plates)

@app.route('/live-detection')
@login_required
def live_detection():
    return render_template('live_detection.html')

@app.route('/search-plates')
@login_required
def search_plates():
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    
    # Build query
    query = DetectedPlate.query.filter_by(user_id=current_user.id)
    
    # Apply filters
    plate_number = request.args.get('plate_number', '').strip()
    if plate_number:
        query = query.filter(DetectedPlate.plate_number.ilike(f'%{plate_number}%'))
    
    start_date = request.args.get('start_date')
    if start_date:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        query = query.filter(DetectedPlate.detected_at >= start_datetime)
    
    end_date = request.args.get('end_date')
    if end_date:
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        query = query.filter(DetectedPlate.detected_at < end_datetime)
    
    min_confidence = request.args.get('min_confidence', type=float)
    if min_confidence:
        query = query.filter(DetectedPlate.confidence >= min_confidence)
    
    # Export functionality
    if request.args.get('export') == 'csv':
        return export_search_results(query)
    
    # Pagination
    pagination = query.order_by(desc(DetectedPlate.detected_at)).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    results = pagination.items
    total_results = pagination.total
    
    return render_template('search_plates.html',
                         results=results,
                         total_results=total_results,
                         pagination=pagination)

@app.route('/plate-history')
@login_required
def plate_history():
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    
    # Build query
    query = DetectedPlate.query.filter_by(user_id=current_user.id)
    
    # Apply filters
    filter_text = request.args.get('filter_text', '').strip()
    if filter_text:
        query = query.filter(DetectedPlate.plate_number.ilike(f'%{filter_text}%'))
    
    # Apply sorting
    sort_by = request.args.get('sort_by', 'detected_at')
    order = request.args.get('order', 'desc')
    
    if sort_by == 'detected_at':
        sort_column = DetectedPlate.detected_at
    elif sort_by == 'plate_number':
        sort_column = DetectedPlate.plate_number
    elif sort_by == 'confidence':
        sort_column = DetectedPlate.confidence
    else:
        sort_column = DetectedPlate.detected_at
    
    if order == 'asc':
        query = query.order_by(asc(sort_column))
    else:
        query = query.order_by(desc(sort_column))
    
    # Pagination
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    plates = pagination.items
    
    # Statistics
    total_plates = DetectedPlate.query.filter_by(user_id=current_user.id).count()
    
    today = datetime.now().date()
    today_plates = DetectedPlate.query.filter(
        and_(DetectedPlate.user_id == current_user.id,
             DetectedPlate.detected_at >= today)
    ).count()
    
    week_ago = today - timedelta(days=7)
    week_plates = DetectedPlate.query.filter(
        and_(DetectedPlate.user_id == current_user.id,
             DetectedPlate.detected_at >= week_ago)
    ).count()
    
    unique_plates = db.session.query(DetectedPlate.plate_number)\
                             .filter(and_(DetectedPlate.user_id == current_user.id,
                                        DetectedPlate.plate_number.isnot(None)))\
                             .distinct().count()
    
    return render_template('plate_history.html',
                         plates=plates,
                         pagination=pagination,
                         total_plates=total_plates,
                         today_plates=today_plates,
                         week_plates=week_plates,
                         unique_plates=unique_plates)

# Helper functions
def export_search_results(query):
    """Export search results as CSV"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Plate Number', 'Confidence', 'Detected At', 'Image Path'])
    
    # Write data
    for plate in query.all():
        writer.writerow([
            plate.id,
            plate.plate_number or 'N/A',
            f"{plate.confidence:.1f}%",
            plate.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
            plate.image_path
        ])
    
    output.seek(0)
    
    # Create response
    response = send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'anpr_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    return response

# SocketIO Events
@socketio.on('start_detection')
@login_required
def handle_start_detection():
    global camera, detection_active, detection_thread

    print("DEBUG: Start detection requested")
    try:
        if not detection_active:
            print("DEBUG: Initializing camera")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

            if not camera.isOpened():
                print("DEBUG: Camera failed to open")
                emit('detection_error', {'error': 'Could not open camera'})
                return

            print("DEBUG: Starting detection thread")
            detection_active = True
            detection_thread = threading.Thread(target=detection_loop)
            detection_thread.daemon = True
            detection_thread.start()

            print("DEBUG: Detection started successfully")
            emit('detection_started', {'status': 'Detection started'})
        else:
            print("DEBUG: Detection already active")
    except Exception as e:
        print(f"DEBUG: Error starting detection: {e}")
        emit('detection_error', {'error': str(e)})

@socketio.on('stop_detection')
@login_required
def handle_stop_detection():
    global camera, detection_active

    detection_active = False
    if camera:
        camera.release()
        camera = None

    emit('detection_stopped', {'status': 'Detection stopped'})

@socketio.on('capture_frame')
@login_required
def handle_capture_frame():
    global camera

    if camera and detection_active:
        ret, frame = camera.read()
        if ret:
            # Process frame for plate detection
            detections = anpr_service.process_frame(frame, current_user.id)

            # Save detections to database
            for detection in detections:
                plate = DetectedPlate(
                    plate_number=detection['plate_text'] if detection['plate_text'] else None,
                    confidence=float(detection['confidence']),
                    image_path=detection['image_path'],
                    user_id=current_user.id,
                    bbox_x=int(detection['bbox'][0]),
                    bbox_y=int(detection['bbox'][1]),
                    bbox_width=int(detection['bbox'][2]),
                    bbox_height=int(detection['bbox'][3])
                )
                db.session.add(plate)

            if detections:
                db.session.commit()

                # Update system stats
                stats = SystemStats.get_stats()
                stats.update_detection_count()

                for detection in detections:
                    if detection['plate_text']:
                        stats.update_ocr_success()

                # Emit capture saved event
                emit('capture_saved', {
                    'detections': len(detections),
                    'image_url': url_for('static', filename=detections[0]['image_path'].replace('static/', '')) if detections else None,
                    'plate_text': detections[0]['plate_text'] if detections else None
                })
            else:
                emit('detection_error', {'error': 'No plates detected in frame'})
    else:
        emit('detection_error', {'error': 'Camera not active'})

def detection_loop():
    """Main detection loop running in separate thread"""
    global camera, detection_active

    print("DEBUG: Detection loop started")

    # Track auto-saves to prevent spam
    last_save_time = {}  # plate_text -> timestamp
    save_count_per_minute = 0
    minute_start = time.time()

    while detection_active and camera:
        print("DEBUG: Detection loop iteration")
        try:
            ret, frame = camera.read()
            if not ret:
                break

            # Detect plates in frame
            detections = anpr_service.detect_plates(frame)

            # Draw detections on frame
            if detections:
                detection_results = []
                saved_detections = []
                current_time = time.time()

                # Reset counter every minute
                if current_time - minute_start > 60:
                    save_count_per_minute = 0
                    minute_start = current_time

                for (x, y, w, h) in detections:
                    plate_roi = anpr_service.extract_plate_roi(frame, x, y, w, h)
                    plate_text, confidence = anpr_service.extract_text_from_plate(plate_roi)

                    detection_result = {
                        'bbox': (x, y, w, h),
                        'plate_text': plate_text,
                        'confidence': confidence,
                        'auto_saved': False
                    }

                    # Auto-save logic - more aggressive to catch failed OCR
                    if Config.AUTO_SAVE_DETECTIONS:
                        # If OCR failed completely, save as unknown for debugging
                        if confidence == 0.0 and not plate_text:
                            plate_text = "UNKNOWN"  # Mark as unknown instead of hardcoded value
                            confidence = 0.0  # Keep original confidence
                            print(f"DEBUG: OCR failed completely, marking as UNKNOWN")

                        should_save = (
                            save_count_per_minute < Config.MAX_DETECTIONS_PER_MINUTE and
                            (
                                (confidence >= Config.MIN_CONFIDENCE_TO_SAVE and plate_text and plate_text != 'Unknown' and len(plate_text.strip()) > 0) or
                                (confidence == 0.0)  # Save even failed OCR attempts for debugging
                            )
                        )

                        # Avoid duplicate saves of same plate within 5 seconds
                        plate_key = plate_text or 'unknown'
                        if should_save and plate_key in last_save_time:
                            if current_time - last_save_time[plate_key] < 5:
                                should_save = False

                        if should_save:
                            try:
                                # Use app context to access database
                                with app.app_context():
                                    # Get the first user (admin) as fallback for auto-save
                                    user = User.query.first()
                                    if user:
                                        # Save plate image
                                        image_path = anpr_service.save_plate_image(plate_roi, user.id)

                                        # Save to database
                                        plate = DetectedPlate(
                                            plate_number=plate_text,
                                            confidence=float(confidence),
                                            image_path=image_path,
                                            user_id=user.id,
                                            bbox_x=int(x),
                                            bbox_y=int(y),
                                            bbox_width=int(w),
                                            bbox_height=int(h)
                                        )
                                        db.session.add(plate)
                                        db.session.commit()

                                        # Update tracking
                                        last_save_time[plate_key] = current_time
                                        save_count_per_minute += 1
                                        detection_result['auto_saved'] = True
                                        saved_detections.append(detection_result)

                                        # Update system stats
                                        stats = SystemStats.get_stats()
                                        stats.update_detection_count()
                                        if plate_text:
                                            stats.update_ocr_success()

                                        print(f"Auto-saved plate: {plate_text} (confidence: {confidence:.1f}%)")

                            except Exception as e:
                                print(f"Auto-save error: {e}")

                    detection_results.append(detection_result)

                # Draw bounding boxes and labels
                frame = anpr_service.draw_detections(frame, detection_results)

                # Convert numpy types to Python native types for JSON serialization
                json_safe_results = []
                for result in detection_results:
                    json_safe_result = {
                        'bbox': [int(result['bbox'][0]), int(result['bbox'][1]),
                                int(result['bbox'][2]), int(result['bbox'][3])],
                        'plate_text': result['plate_text'],
                        'confidence': float(result['confidence']),
                        'auto_saved': result.get('auto_saved', False)
                    }
                    json_safe_results.append(json_safe_result)

                # Emit detection results
                socketio.emit('detection_result', {
                    'detections': json_safe_results,
                    'count': len(json_safe_results),
                    'auto_saved_count': len(saved_detections)
                })

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Emit frame to client
            socketio.emit('video_frame', {'frame': frame_data})

            # Small delay to prevent overwhelming the client
            time.sleep(0.1)

        except Exception as e:
            socketio.emit('detection_error', {'error': str(e)})
            break

    # Cleanup
    if camera:
        camera.release()

# API Routes
@app.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    stats = SystemStats.get_stats()
    user_detections = DetectedPlate.query.filter_by(user_id=current_user.id).count()

    total_with_text = DetectedPlate.query.filter(DetectedPlate.plate_number.isnot(None)).count()
    accuracy = (total_with_text / stats.total_detections * 100) if stats.total_detections > 0 else 0

    return jsonify({
        'success': True,
        'stats': {
            'total_detections': stats.total_detections,
            'successful_ocr': stats.successful_ocr,
            'user_detections': user_detections,
            'accuracy': round(accuracy, 1)
        }
    })

@app.route('/api/delete-plates', methods=['POST'])
@login_required
def delete_plates():
    try:
        data = request.get_json()
        plate_ids = data.get('plate_ids', [])

        # Delete plates belonging to current user
        deleted_count = DetectedPlate.query.filter(
            and_(DetectedPlate.id.in_(plate_ids),
                 DetectedPlate.user_id == current_user.id)
        ).delete(synchronize_session=False)

        db.session.commit()

        return jsonify({
            'success': True,
            'deleted_count': deleted_count
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export-history')
@login_required
def export_history():
    plate_ids = request.args.get('plate_ids', '').split(',') if request.args.get('plate_ids') else None

    query = DetectedPlate.query.filter_by(user_id=current_user.id)

    if plate_ids and plate_ids[0]:  # Check if plate_ids is not empty
        query = query.filter(DetectedPlate.id.in_(plate_ids))

    return export_search_results(query)

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        # Delete all plates for current user
        deleted_count = DetectedPlate.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()

        return jsonify({
            'success': True,
            'deleted_count': deleted_count
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recent-detections')
@login_required
def api_recent_detections():
    """Get recent detections for the current user"""
    try:
        recent_plates = DetectedPlate.query.filter_by(user_id=current_user.id)\
                                         .order_by(desc(DetectedPlate.detected_at))\
                                         .limit(10).all()

        detections = []
        for plate in recent_plates:
            detections.append({
                'id': plate.id,
                'plate_number': plate.plate_number,
                'confidence': plate.confidence,
                'detected_at': plate.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
                'image_url': url_for('static', filename=plate.image_path.replace('static/', '')) if plate.image_path else None
            })

        return jsonify({
            'success': True,
            'detections': detections
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-detection')
@login_required
def api_test_detection():
    """Create a test detection to verify the system is working"""
    try:
        # Create a test detection with the visible plate number
        test_plate = DetectedPlate(
            plate_number='RJ14CV0002',
            confidence=95.0,
            image_path='static/uploads/test_plate.jpg',
            user_id=current_user.id,
            bbox_x=100,
            bbox_y=100,
            bbox_width=200,
            bbox_height=50
        )
        db.session.add(test_plate)
        db.session.commit()

        # Update system stats
        stats = SystemStats.get_stats()
        stats.update_detection_count()
        stats.update_ocr_success()

        return jsonify({
            'success': True,
            'message': 'Test detection created successfully',
            'plate_number': 'RJ14CV0002'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # Create default admin user if no users exist
        if User.query.count() == 0:
            admin = User(username='admin', email='admin@anpr.com')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
