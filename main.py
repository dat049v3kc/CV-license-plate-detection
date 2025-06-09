import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load models
plate_model = YOLO('best_most.pt')
char_model = YOLO('kytu.pt')

def detect_license_plates(image):
    results = plate_model(image)
    return results

def detect_characters(image):
    results = char_model(image)
    return results

def clahe_enhancement(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if image has 3 channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If grayscale already, don't convert
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    return enhanced_image

def clean_text(text):
    # Remove any characters that are not alphanumeric
    return re.sub(r'[^A-Z0-9]', '', text)

def process_image(image):
    plate_results = detect_license_plates(image)
    detected_texts = []  # List to collect detected texts
    for result in plate_results:
        for plate in result.boxes.data:
            x1, y1, x2, y2 = map(int, plate[:4])
            plate_crop = image[y1:y2, x1:x2]

            # Apply CLAHE for enhancement
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe_enhancement(gray)

            # Use Tesseract for OCR
            text = pytesseract.image_to_string(enhanced_gray, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            cleaned_text = clean_text(text)
            detected_texts.append(cleaned_text)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, cleaned_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image, detected_texts

def save_detected_plates(detected_texts):
    for char_text in detected_texts:
        if isinstance(char_text, str) and char_text.strip():  # Ensure char_text is a non-empty string
            with open("detected_license_plates.txt", "a") as file:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"{char_text}\t{current_datetime}\n")

# Streamlit interface
st.title("ỨNG DỤNG PHÁT HIỆN VÀ NHẬN DIỆN BIỂN SỐ XE")

# Tạo hai lựa chọn cho người dùng
option = st.selectbox("Chọn loại tệp để tải lên", ("Hình ảnh", "Video"))

if option == "Hình ảnh":
    uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Xử lý hình ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        processed_image, detected_texts = process_image(image)

        # Hiển thị hình ảnh đã xử lý
        st.image(processed_image, channels="BGR")

        # Lưu các biển số xe nhận diện được
        save_detected_plates(detected_texts)
        st.success("Nhận diện hình ảnh thành công!")

elif option == "Video":
    uploaded_file = st.file_uploader("Chọn một video...", type=["mp4", "mov"])
    if uploaded_file is not None:
        # Xử lý video
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture("temp_video.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, detected_texts = process_image(frame)
            st.image(processed_frame, channels="BGR")

        cap.release()

        # Lưu các biển số xe nhận diện được
        save_detected_plates(detected_texts)
        st.success("Nhận diện video thành công!")
