# ANPR Web Application

A comprehensive Automatic Number Plate Recognition (ANPR) web application built with Flask, OpenCV, and modern web technologies.

## Features

- 🔐 **User Authentication** - Secure login and registration system
- 📹 **Live Detection** - Real-time number plate detection using webcam
- 🔍 **OCR Text Extraction** - Extract text from detected plates using Tesseract
- 📊 **Dashboard** - Statistics and recent detections overview
- 🔎 **Advanced Search** - Search plates by number, date range, and confidence
- 📈 **History Management** - View, filter, and export detection history
- 💾 **Database Storage** - Store all detections with timestamps
- 📱 **Responsive Design** - Works on desktop and mobile devices
- 🎨 **Modern UI** - Beautiful interface with Bootstrap 5

## Technology Stack

- **Backend**: Flask, SQLAlchemy, Flask-SocketIO
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Computer Vision**: OpenCV, Tesseract OCR
- **Database**: SQLite (easily upgradeable to PostgreSQL)
- **Real-time**: WebSocket for live camera feed

## Installation

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** installed:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd "C:\PYTHON 2MINS\ANPR_with_opencv"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Tesseract path** (if needed)
   Edit `config.py` and update the `TESSERACT_CMD` path:
   ```python
   TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
   # TESSERACT_CMD = '/usr/bin/tesseract'  # Linux/Mac
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and go to: `http://localhost:5000`

## Default Login

- **Username**: `admin`
- **Password**: `admin123`

## Usage

### 1. Dashboard
- View system statistics and recent detections
- Quick access to all features

### 2. Live Detection
- Click "Start Detection" to begin real-time plate recognition
- Use "Capture" button to save detected plates
- View live detection results in the sidebar

### 3. Search Plates
- Search by plate number (partial matches supported)
- Filter by date range
- Set minimum confidence threshold
- Export search results as CSV

### 4. History Management
- View all detected plates with pagination
- Sort by date, plate number, or confidence
- Bulk delete operations
- Export history as CSV

## File Structure

```
ANPR_with_opencv/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── models.py                       # Database models
├── anpr_service.py                 # ANPR detection service
├── requirements.txt                # Python dependencies
├── haarcascade_russian_plate_number.xml  # Haar cascade for plate detection
├── templates/                      # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── live_detection.html
│   ├── search_plates.html
│   └── plate_history.html
├── static/                         # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── uploads/                    # Saved plate images
└── IMAGES/                         # Original detection images
```

## Configuration

### Camera Settings
Edit `config.py` to adjust camera parameters:
```python
FRAME_WIDTH = 1000
FRAME_HEIGHT = 480
MIN_PLATE_AREA = 500
```

### Database
The application uses SQLite by default. To use PostgreSQL:
```python
SQLALCHEMY_DATABASE_URI = 'postgresql://user:password@localhost/anpr_db'
```

## API Endpoints

- `GET /api/dashboard-stats` - Get dashboard statistics
- `POST /api/delete-plates` - Delete selected plates
- `GET /api/export-history` - Export detection history
- `POST /api/clear-history` - Clear all detection history

## WebSocket Events

- `start_detection` - Start camera detection
- `stop_detection` - Stop camera detection
- `capture_frame` - Capture and save current frame
- `video_frame` - Receive video frames
- `detection_result` - Receive detection results

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Check camera permissions in your browser
- Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Tesseract Issues
- Verify Tesseract is installed correctly
- Check the path in `config.py`
- Install additional language packs if needed

### Performance Issues
- Reduce frame size in config
- Increase detection threshold
- Close other applications using the camera

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure you comply with local laws regarding surveillance and privacy.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Ensure all dependencies are installed correctly

## Future Enhancements

- [ ] Multiple camera support
- [ ] Advanced OCR models
- [ ] Real-time alerts
- [ ] Mobile app
- [ ] Cloud deployment
- [ ] Advanced analytics
