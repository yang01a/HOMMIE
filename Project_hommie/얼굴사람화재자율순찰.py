from flask import Flask, render_template, Response, request, redirect, url_for
import threading
import json
import numpy as np
import face_recognition
import os
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cvzone
import math
from ultralytics import YOLO
import subprocess
import gc

app = Flask(__name__)

# Initialize ROS node
rospy.init_node('face_recognition_web', anonymous=True)

# Initialize ROS bridge
bridge = CvBridge()

# Initialize global variables for face recognition and fire detection
FACE_DATA_PATH = "/home/yang/Desktop/family/list/face_data.json"
known_face_encodings = []
known_face_names = []

# Initialize flags for toggling features
face_recognition_enabled = False
fire_detection_enabled = False

# Global variable for the current frame
frame = None

# Load YOLO models
fire_model = YOLO('/home/yang/Desktop/family/fire_model.pt')
person_model = YOLO('/home/yang/Desktop/family/yolov5n.pt')  # YOLOv5n model for person detection

classnames = ['fire']

# Initialize paths for patrol
MAP_FILE = os.path.expanduser("~/mmmap.yaml")  # Path to the map file
RANDOM_PATROL_PATH = os.path.expanduser("~/catkin_ws/src/random_patrol_package/src/random_patrol.py")  # Path to the patrol script

def load_face_data():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if os.path.exists(FACE_DATA_PATH):
        with open(FACE_DATA_PATH, "r") as f:
            try:
                face_data = json.load(f)
                known_face_names = face_data["names"]
                known_face_encodings = [np.array(encoding) for encoding in face_data["encodings"]]
                print("Face data loaded.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {FACE_DATA_PATH}. Initializing with empty data.")
                known_face_names = []
                known_face_encodings = []
    else:
        print(f"Face data file not found at {FACE_DATA_PATH}. Starting with an empty list.")

def save_face_data():
    face_data = {
        "names": known_face_names,
        "encodings": [encoding.tolist() for encoding in known_face_encodings]
    }
    with open(FACE_DATA_PATH, "w") as f:
        json.dump(face_data, f)
    print("Face data saved.")

def augment_face_encoding(face_encoding):
    """Apply slight variations to the face encoding for data augmentation."""
    noise = np.random.normal(0, 0.01, face_encoding.shape)  # Add small noise
    return face_encoding + noise

def augment_face_image(image):
    """Generate augmented versions of the input face image."""
    augmented_images = []

    # Rotate image
    rows, cols, _ = image.shape
    for angle in [-10, 10]:  # Rotate by -10 and 10 degrees
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)

    # Adjust brightness
    for alpha in [0.8, 1.2]:  # Dim and brighten
        brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(brightened)

    # Scale image
    for scale in [0.9, 1.1]:  # Slightly zoom in and out
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        augmented_images.append(scaled)

    return augmented_images

def register_face(current_frame, user_name):
    global known_face_encodings, known_face_names
    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Save original encoding
            known_face_encodings.append(face_encoding)
            known_face_names.append(user_name)

            # Crop and augment face image
            face_image = rgb_frame[top:bottom, left:right]
            augmented_images = augment_face_image(face_image)

            for aug_image in augmented_images:
                aug_face_encodings = face_recognition.face_encodings(aug_image)
                for aug_face_encoding in aug_face_encodings:
                    augmented_encoding = augment_face_encoding(aug_face_encoding)
                    known_face_encodings.append(augmented_encoding)
                    known_face_names.append(user_name)

        save_face_data()
        print(f"Face registered for {user_name} with data augmentation.")
    else:
        print("No face detected.")

# ROS image callback to capture and process image
def image_callback(msg):
    global frame
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS image message to OpenCV format

# Subscribe to the camera topic
rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)

# Recognize faces function
def recognize_faces(frame):
    global known_face_encodings, known_face_names
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
        if best_match_index >= 0 and face_distances[best_match_index] < 0.4:
            face_names.append(known_face_names[best_match_index])
        else:
            face_names.append("Unknown")

    return face_locations, face_names

# Fire detection function
def detect_fire(frame):
    result = fire_model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 70:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
    return frame

# Person detection function
def detect_person(frame):
    results = person_model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if Class == 0 and confidence > 70:  # Class 0 is 'person' in COCO
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {confidence}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

# Function to generate MJPEG stream
def gen_frames():
    global face_recognition_enabled, fire_detection_enabled, frame

    while not rospy.is_shutdown():
        if frame is not None:
            processed_frame = frame.copy()
            
            if face_recognition_enabled:
                face_locations, face_names = recognize_faces(processed_frame)
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # "Unknown" 얼굴만 빨간색으로 표시
                    if name == "Unknown":
                        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 0, 255), 2)  # 빨간색 바운딩 박스
                        cv2.putText(processed_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # 빨간색 텍스트
                    else:
                        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 녹색 바운딩 박스
                        cv2.putText(processed_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 녹색 텍스트

            if fire_detection_enabled:
                processed_frame = detect_fire(processed_frame)
            
            processed_frame = detect_person(processed_frame)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

        gc.collect()  # Memory management for garbage collection

@app.route('/')
def index():
    return render_template('index.html')  # This HTML will include the new patrol button

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_face', methods=['POST'])
def register_face_route():
    user_name = request.form.get('name')
    if frame is not None and user_name:
        register_face(frame, user_name)
        return redirect(url_for('index'))
    return "Error: No frame available or name not provided.", 400

@app.route('/toggle_face')
def toggle_face():
    global face_recognition_enabled
    face_recognition_enabled = not face_recognition_enabled
    status = "enabled" if face_recognition_enabled else "disabled"
    return f"Face recognition {status}."

@app.route('/toggle_fire')
def toggle_fire():
    global fire_detection_enabled
    fire_detection_enabled = not fire_detection_enabled
    status = "enabled" if fire_detection_enabled else "disabled"
    return f"Fire detection {status}."

import signal

# Global variable to store the patrol processes
patrol_process = None
launch_process = None

@app.route('/start_patrol', methods=['POST'])
def start_patrol():
    global patrol_process, launch_process

    try:
        # ROS 환경 변수 설정
        env = os.environ.copy()
        env["ROS_MASTER_URI"] = "http://192.168.123.8:11311"  # ROS Master URI 설정
        env["ROS_PACKAGE_PATH"] = "/opt/ros/noetic/share"     # ROS Package 경로 설정

        # roslaunch 실행 (맵 로딩)
        launch_process = subprocess.Popen(
            ["roslaunch", "turtlebot3_navigation", "turtlebot3_navigation.launch", f"map_file:={MAP_FILE}"], env=env
        )

        # rosrun 실행 (순찰 시작)
        patrol_process = subprocess.Popen(["python3", RANDOM_PATROL_PATH], env=env)

        return "순찰이 시작되었습니다. 로봇이 자율 순찰을 수행 중입니다."
    except Exception as e:
        return f"오류 발생: {str(e)}"

@app.route('/stop_patrol', methods=['POST'])
def stop_patrol():
    global patrol_process, launch_process

    try:
        # If patrol_process is running, terminate it
        if patrol_process is not None:
            patrol_process.terminate()  # Attempt to terminate the patrol process
            patrol_process.wait()       # Wait for it to properly terminate

        # If launch_process is running, terminate it
        if launch_process is not None:
            launch_process.terminate()  # Attempt to terminate the launch process
            launch_process.wait()       # Wait for it to properly terminate

        # Reset process references
        patrol_process = None
        launch_process = None

        return "순찰이 중지되었습니다."

    except Exception as e:
        return f"오류 발생: {str(e)}"


if __name__ == '__main__':
    load_face_data()
    app.run(host='0.0.0.0', port=5000, threaded=True)

