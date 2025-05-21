import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import av
import cv2
import numpy as np
import threading
from PIL import Image
from torchvision import transforms
from model import SwinWithSE, SEBlock
import mediapipe as mp
import time
import matplotlib.pyplot as plt

# Tell PyTorch it’s safe to unpickle SwinWithSE
# torch.serialization.add_safe_globals({'SwinWithSE': SwinWithSE})

model =SwinWithSE(7)
criterion =torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('swin_with_se_fer2013_full.pth', map_location=torch.device(device), weights_only=False)
# print(model)
    # Preprocessing transform (match your training pipeline)
preprocess =  transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emoji_path = './emoji/'


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def process_frame(frame):
    emotion = 'neutral'
    emoji_path = './emoji/'+emotion+'.png'
    # Convert BGR (OpenCV) to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MediaPipe
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            w, h = min(w, iw-x), min(h, ih-y)
            
            # Extract face region
            face_roi = frame_rgb[y:y+h, x:x+w]
            if face_roi.size == 0:  # Skip if ROI is empty
                continue
            
            # Preprocess face for model
            face_pil = Image.fromarray(face_roi)
            face_tensor = preprocess(face_pil).unsqueeze(0).to(device)
            
            # Predict emotion
            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotion_labels[predicted.item()]

                emoji_path = './emoji/' + emotion + '.png'
    return emotion, emoji_path

st.set_page_config(layout="wide")
st.title("Real-time Webcam and Periodic Capture")

# Shared state to hold the latest captured frame
captured_frame = None
lock = threading.Lock()

# Video frame callback to process each frame from webcam
def video_frame_callback(frame):
    global captured_frame, lock
    img = frame.to_ndarray(format="bgr24")

    # Every 5 seconds, update the captured_frame
    current_time = time.time()
    if not hasattr(video_frame_callback, "last_capture_time"):
        video_frame_callback.last_capture_time = 0

    if current_time - video_frame_callback.last_capture_time > 2:
        with lock:
            captured_frame = img.copy()
        video_frame_callback.last_capture_time = current_time

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Live Webcam Feed")
    ctx = webrtc_streamer(
        key="webcam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.header("Captured Image (updates every 5 seconds)")
    image_placeholder = st.empty()

    # Update the captured image every second to show latest capture
    while True:
        with lock:
            frame = captured_frame

        if frame is not None:
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, emoji_path  = process_frame(img_rgb)
            emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
            image_placeholder.image(emoji, caption="Captured Emotion", use_container_width=True)
        else:
            image_placeholder.text("Waiting for capture...")

        time.sleep(1)
