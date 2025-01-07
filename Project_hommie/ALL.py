import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import face_recognition
import torch
import numpy as np
from yolov5.utils.general import non_max_suppression
from yolov5.models.common import DetectMultiBackend
import os
import json
from PIL import Image as PILImage, ImageEnhance
import random
import sys
import threading
import tkinter as tk
from tkinter import simpledialog
from PIL import ImageTk

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# YOLOv5 경로 설정
yolov5_path = "/home/yang/Desktop/family/yolov5"
MODEL_PATH = "/home/yang/Desktop/family/yolov5/yolov5n.pt"  # YOLOv5 모델 경로
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 디렉터리
DATASET_DIR = "/home/yang/Desktop/family/dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# YOLOv5 모델 로드
model = DetectMultiBackend(MODEL_PATH, device=DEVICE)

# 얼굴 등록용 변수
known_face_encodings = []
known_face_names = []

# 얼굴 데이터 저장 경로
FACE_DATA_PATH = "/home/yang/Desktop/family/list/face_data.json"

# 얼굴 데이터를 저장하는 함수
def save_face_data():
    face_data = {
        "names": known_face_names,
        "encodings": [encoding.tolist() for encoding in known_face_encodings]
    }
    with open(FACE_DATA_PATH, "w") as f:
        json.dump(face_data, f)
    print("Face data saved.")

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
                # 초기화된 데이터로 빈 리스트 반환
                known_face_names = []
                known_face_encodings = []
    else:
        print(f"Face data file not found at {FACE_DATA_PATH}. Starting with an empty list.")


# 데이터 증폭 함수
def augment_image(face_crop):
    augmented_images = []
    try:
        # 1. 회전 (0, 90, 180, 270도)
        for angle in [0, 90, 180, 270]:
            rotated = PILImage.fromarray(face_crop).rotate(angle)
            augmented_images.append(np.array(rotated))

        # 2. 밝기 조정
        enhancer = ImageEnhance.Brightness(PILImage.fromarray(face_crop))
        bright_image = enhancer.enhance(1.5)  # 밝기를 1.5배로 증가
        augmented_images.append(np.array(bright_image))

        # 3. 대비 조정
        enhancer = ImageEnhance.Contrast(PILImage.fromarray(face_crop))
        contrast_image = enhancer.enhance(1.5)  # 대비를 1.5배로 증가
        augmented_images.append(np.array(contrast_image))

        # 4. 랜덤 크기 변경 (zoom in/out)
        h, w = face_crop.shape[:2]
        scale_factor = random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized_image = cv2.resize(face_crop, (new_w, new_h))
        augmented_images.append(resized_image)

        # 증폭된 이미지 수가 5개가 되도록 할 수 있습니다.
        while len(augmented_images) < 5:
            augmented_images.append(face_crop)  # 기본 이미지를 추가하여 5개로 맞추기

    except Exception as e:
        print(f"Data augmentation error: {e}")
    return augmented_images

# 얼굴 등록 함수
def register_face():
    global known_face_encodings, known_face_names, frame, detected_faces
    for (x1, y1, x2, y2) in detected_faces:
        face_crop = frame[y1:y2, x1:x2]
        rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_face_crop)
        
        if face_encoding:
            print(f"Face encoding found: {face_encoding}")
            face_encoding = face_encoding[0]
            user_name = simpledialog.askstring("Input", "Enter your name:")

            # 데이터 증폭과 등록
            augmented_images = augment_image(face_crop)
            all_images = [face_crop] + augmented_images  # 원본 + 증폭된 이미지들

            for aug_img in all_images:
                aug_rgb_face_crop = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                aug_face_encoding = face_recognition.face_encodings(aug_rgb_face_crop)
                if aug_face_encoding:
                    print(f"Augmented face encoding found: {aug_face_encoding}")
                    known_face_encodings.append(aug_face_encoding[0])
                    known_face_names.append(user_name)

            print(f"Registered {user_name}'s face with augmented images")
            save_face_data()
        else:
            print("No face encoding found.")



# YOLOv5로 얼굴 검출 함수
def detect_faces(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    pred = model(img, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
    return pred

# 얼굴 인식 및 출석 비교 함수
def recognize_faces(frame):
    global known_face_encodings, known_face_names
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    if not known_face_encodings:  # 등록된 얼굴 데이터가 없는 경우
        print("No known faces to compare.")
        return face_locations, ["Unknown"] * len(face_encodings)

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) == 0:  # 비교 대상이 없을 때
            face_names.append("Unknown")
        else:
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            if face_distances[best_match_index] < 0.4:  # 허용 오차
                name = known_face_names[best_match_index]
            face_names.append(name)

        # 디버깅 출력
        print(f"Distances: {face_distances}")
        print(f"Best match index: {best_match_index if len(face_distances) > 0 else 'N/A'}, Name: {name}")

    return face_locations, face_names



# ROS 이미지 콜백 함수
def image_callback(msg):
    global frame, detected_faces
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")

    detections = detect_faces(frame)
    detected_faces = []
    
    if detections[0] is not None:
        for *xyxy, conf, cls in detections[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            detected_faces.append((x1, y1, x2, y2))

    face_locations, face_names = recognize_faces(frame)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # OpenCV 이미지를 PIL로 변환하여 Tkinter Label에 업데이트
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Tkinter Label에 이미지 업데이트
    label.img = img_tk
    label.configure(image=img_tk)

# Tkinter GUI 생성
def gui_thread():
    global label
    root = tk.Tk()
    root.title("Face Recognition")

    # Tkinter Label 위젯을 사용하여 웹캠 영상 표시
    label = tk.Label(root)
    label.pack(padx=10, pady=10)

    # 얼굴 등록 버튼
    button = tk.Button(root, text="your face", command=register_face)
    button.pack(pady=20)

    root.mainloop()

# 메인 함수
def main():
    load_face_data()
    
    # Tkinter GUI를 별도의 스레드로 실행
    threading.Thread(target=gui_thread, daemon=True).start()

    rospy.init_node('face_recognition_node')
    rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
    
    print("Starting face recognition. Press Ctrl+C to quit.")
    rospy.spin()

if __name__ == "__main__":
    main()
