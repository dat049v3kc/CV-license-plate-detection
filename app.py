import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load models
plate_model = YOLO('best_most.pt')
char_model = YOLO('best.pt')

def detect_license_plates(image):
    results = plate_model.predict(image)
    return results

def detect_characters(image):
    results = char_model.predict(image)
    return results

def clahe_enhancement(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Kiểm tra nếu ảnh có 3 kênh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Nếu ảnh đã là grayscale, không cần chuyển đổi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    return enhanced_image

def canny_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def hough_transform(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def affine_transform(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst

def process_image(image):
    plate_results = detect_license_plates(image)
    detected_texts = []  # List to collect detected texts
    for result in plate_results:
        for plate in result.boxes.data:
            x1, y1, x2, y2 = map(int, plate[:4])
            plate_crop = image[y1:y2, x1:x2]

            # Apply image processing techniques to improve OCR results
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = clahe_enhancement(gray)
            edges = canny_edge_detection(gray)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Chuyển về ảnh màu để vẽ lên
            hough_img = hough_transform(np.copy(edges))
            affine_img = affine_transform(np.copy(hough_img))

            # Ensure the image has three channels
            if len(affine_img.shape) == 2:  # Nếu là ảnh grayscale
                color_crop = cv2.cvtColor(affine_img, cv2.COLOR_GRAY2BGR)
            else:
                color_crop = affine_img

            # Detect characters in the plate
            char_results = detect_characters(color_crop)
            char_text = ""
            for char_result in char_results:
                for char in char_result.boxes.data:
                    char_x1, char_y1, char_x2, char_y2 = map(int, char[:4])
                    if 0 <= char_x1 < char_x2 <= color_crop.shape[1] and 0 <= char_y1 < char_y2 <= color_crop.shape[0]:
                        char_crop = color_crop[char_y1:char_y2, char_x1:char_x2]
                        text = pytesseract.image_to_string(char_crop,
                                                           config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                        char_text += text.strip()

            print("Detected License Plate:", char_text)
            detected_texts.append(char_text)  # Append detected text to the list
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, char_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
        save_detected_plstates(detected_texts)
        st.success("Nhận diện video thành công!")
