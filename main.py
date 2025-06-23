from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from inf_video import predict_frames_in_directory, predict_from_two_directories
import shutil
import tempfile
import os
import logging
import cv2
import re
import numpy as np
from ultralytics import YOLO
import unicodedata

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)




# Utility functions
def blurred_frame_differencing(frame1, frame2):
    blurred_frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    blurred_frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    return cv2.absdiff(blurred_frame1, blurred_frame2)

def extract_nostrils_from_video(video_path, input_dir, temp_dir):
    model = YOLO('nostrils_detector.pt')
    roi_dir = os.path.join(temp_dir, 'roi')
    os.makedirs(roi_dir, exist_ok=True)

    for frame_file in os.listdir(input_dir):
        frame_path = os.path.join(input_dir, frame_file)
        process_frame(model, frame_path, roi_dir)

    return roi_dir

def process_frame(model, frame_path, roi_dir):
    img = cv2.imread(frame_path)
    results = model(img)
    boxes = results[0].boxes.xyxy.tolist()
    for idx, (box) in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_img = img[y1:y2, x1:x2]
        output = os.path.join(roi_dir, frame_path.split('/')[-1])
        cv2.imwrite(output, cropped_img)


def subtract(video_path, input_dir, temp_dir):
    output_dir = os.path.join(temp_dir, 'subtracted_frames')
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.jpg')],
        key=lambda f: int(re.findall(r'\d+', f)[0])
    )

    for i in range(len(frame_files) - 1):
        frame1_path = os.path.join(input_dir, frame_files[i])
        frame2_path = os.path.join(input_dir, frame_files[i + 1])
        subtract_frames(frame1_path, frame2_path, output_dir, i)

    return output_dir

def subtract_frames(frame1_path, frame2_path, output_dir, idx):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    if frame1.shape != frame2.shape:
        if frame1.size < frame2.size:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        else:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    subtracted_frame = blurred_frame_differencing(frame1, frame2)
    output_frame_path = os.path.join(output_dir, f'frame_{idx}.jpg')
    cv2.imwrite(output_frame_path, subtracted_frame)

def secure_filename(filename):
    filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    filename = re.sub(r"[^A-Za-z0-9_.-]", "", filename)
    return filename

def process_video(video_path, extract_nostrils, temp_dir, return_dir=False):
    output_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    count = 0
    frame_skip = 5
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                raise ValueError("Captured frame is None")
            if count % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_path = os.path.join(output_dir, f'frame_{count}.jpg')
                cv2.imwrite(frame_path, gray_frame)
            count += 1
    finally:
        cap.release()

    if extract_nostrils:
        roi_dir = extract_nostrils_from_video(video_path, output_dir, temp_dir)
        subtracted_dir = subtract(video_path, roi_dir, temp_dir)
    else:
        subtracted_dir = subtract(video_path, output_dir, temp_dir)

    if return_dir:
        return subtracted_dir

    model_dir = 'models/best_nostrils_model.keras' if extract_nostrils else 'models/best_abdomen_model.keras'
    result = predict_frames_in_directory(subtracted_dir, model_dir)
    return result

@app.post("/predict")
async def predict(
    video1: UploadFile = File(...),
    video2: UploadFile = File(None),
    model_type: str = Form(...)
):
    with tempfile.TemporaryDirectory() as temp_dir:
        path1 = os.path.join(temp_dir, secure_filename(video1.filename))
        with open(path1, "wb") as f:
            shutil.copyfileobj(video1.file, f)

        if model_type == "Both":
            if not video2:
                return {"error": "Both videos are required for the combined model."}

            path2 = os.path.join(temp_dir, secure_filename(video2.filename))
            with open(path2, "wb") as f:
                shutil.copyfileobj(video2.file, f)

            nostrils_dir = process_video(path1, extract_nostrils=True, temp_dir=temp_dir, return_dir=True)
            abdomen_dir = process_video(path2, extract_nostrils=False, temp_dir=temp_dir, return_dir=True)
            result = predict_from_two_directories(nostrils_dir, abdomen_dir, 'models/best_both_model.keras')

        elif model_type == "Nostrils":
            result = process_video(path1, extract_nostrils=True, temp_dir=temp_dir)

        elif model_type == "Abdomen":
            result = process_video(path1, extract_nostrils=False, temp_dir=temp_dir)

        else:
            return {"error": f"Invalid model_type '{model_type}'. Choose 'Nostrils', 'Abdomen', or 'Both'."}

        return {"result": result}




