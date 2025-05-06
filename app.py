from flask import Flask, render_template, request, Response, redirect, url_for
import os
import cv2
from detector import ObjectDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
video_source = None
detector = ObjectDetector("runs/detect/train/weights/best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_source
    f = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(video_path)
    video_source = video_path
    return redirect(url_for('index'))

@app.route('/camera', methods=['POST'])
def use_camera():
    global video_source
    video_source = 0  # webcam
    return redirect(url_for('index'))

def gen():
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = detector.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return detector.get_stats()

if __name__ == '__main__':
    app.run(debug=True)
