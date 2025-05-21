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

def overlay_emoji(frame, emoji, x=None, y=10, scale=0.1):
    """
    Overlay emoji PNG (with alpha) on BGR frame.
    Position is top-right corner by default (x calculated).
    scale: relative width of emoji to frame width.
    """
    frame_h, frame_w = frame.shape[:2]

    # Resize emoji
    emoji_w = int(frame_w * scale)
    emoji_h = int(emoji.shape[0] * emoji_w / emoji.shape[1])
    emoji_resized = cv2.resize(emoji, (emoji_w, emoji_h), interpolation=cv2.INTER_AREA)

    # Calculate x if not provided (top-right corner)
    if x is None:
        x = frame_w - emoji_w - 10  # 10 px padding from right edge

    y1, y2 = y, y + emoji_h
    x1, x2 = x, x + emoji_w

    # Boundary check
    if y2 > frame_h or x1 < 0:
        return frame  # skip overlay if out of bounds

    # Separate color and alpha channels
    emoji_rgb = emoji_resized[:, :, :3]
    alpha_mask = emoji_resized[:, :, 3] / 255.0

    roi = frame[y1:y2, x1:x2]

    # Blend emoji with ROI using alpha mask
    for c in range(3):
        roi[:, :, c] = (alpha_mask * emoji_rgb[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

    frame[y1:y2, x1:x2] = roi
    return frame


def video_frame_callback(frame):
    global captured_frame, lock
    img = frame.to_ndarray(format="bgr24")

    current_time = time.time()
    if not hasattr(video_frame_callback, "last_capture_time"):
        video_frame_callback.last_capture_time = 0
        video_frame_callback.current_emoji_path = None

    # Only update emoji path every 2 seconds
    if current_time - video_frame_callback.last_capture_time > 1:
        # Call your function that returns emoji path based on current frame
        _,video_frame_callback.current_emoji_path = process_frame(img)
        video_frame_callback.last_capture_time = current_time

        # Optionally update captured_frame for display or other uses
        with lock:
            captured_frame = img.copy()

    # Load emoji image once per frame from the cached path
    emoji = cv2.imread(video_frame_callback.current_emoji_path, cv2.IMREAD_UNCHANGED) if video_frame_callback.current_emoji_path else None

    if emoji is not None:
        img_with_emoji = overlay_emoji(img, emoji, y=10, scale=0.1)
    else:
        img_with_emoji = img

    return av.VideoFrame.from_ndarray(img_with_emoji, format="bgr24")





# Display only one column with the webcam feed and emoji overlay
webrtc_streamer(
    key="webcam_with_emoji",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)