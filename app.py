import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit app
st.title("Object Detection with YOLOv8")
st.subheader("Upload an image to detect objects")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model.predict(img_array)

    # Display results
    st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        frame_with_boxes = results[0].plot()
        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

    cap.release()

from io import BytesIO

# Save annotated image
if st.button("Download Results"):
    output_image = Image.fromarray(results[0].plot())
    buf = BytesIO()
    output_image.save(buf, format="PNG")
    st.download_button("Download", buf.getvalue(), "detection_results.png", "image/png")
