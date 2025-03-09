import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
from PIL import Image
import io
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import base64  # สำหรับแปลงรูปภาพเป็น base64 string ใน HTML

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Face Age Gender Detection",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# สร้าง CSS สวยๆ
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem !important;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .face-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
    }
    .face-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .camera-options {
        margin-top: 10px;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    .camera-mode-selector {
        margin-top: 10px;
        margin-bottom: 15px;
        padding: 15px;
        background-color: #e8f5e9;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
    }
    .mode-info {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# แสดงหัวเว็บ
st.markdown("<h1 class='main-title'>👤 ระบบตรวจจับใบหน้า เพศ และอายุ</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>อัปโหลดรูปภาพหรือใช้กล้องเว็บแคมเพื่อวิเคราะห์</p>", unsafe_allow_html=True)

# นิยามฟังก์ชัน lrn (Local Response Normalization)
@st.cache_resource
def create_lrn_function():
    def lrn(x, depth_radius=5, bias=1, alpha=1, beta=0.5):
        return tf.nn.local_response_normalization(
            x, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta
        )
    return lrn

lrn = create_lrn_function()

# โหลดโมเดลและตั้งค่า (ใช้ cache เพื่อไม่ต้องโหลดใหม่ทุกครั้ง)
@st.cache_resource
def load_models():
    # ตรวจสอบโฟลเดอร์ models ถ้าไม่มีให้สร้าง
    os.makedirs("models", exist_ok=True)
    
    # กำหนดพาธของโมเดล
    face_model_path = "opencv_face_detector_uint8.pb"
    face_proto_path = "opencv_face_detector.pbtxt"
    age_model_path = "age_model_improved.h5"
    gender_model_path = "gender_model_improved.h5"
    
    # ตรวจสอบว่าโมเดลมีอยู่หรือไม่ ถ้าไม่มีให้แสดงข้อความแจ้งเตือน
    models_exist = all([os.path.exists(p) for p in [face_model_path, face_proto_path, age_model_path, gender_model_path]])
    
    if not models_exist:
        # ลองหาโมเดลในไดเร็กทอรีปัจจุบัน
        current_dir_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        model_files = {
            'face_model': [f for f in current_dir_files if f.endswith('_uint8.pb')],
            'face_proto': [f for f in current_dir_files if f.endswith('.pbtxt')],
            'age_model': [f for f in current_dir_files if f.startswith('age') and (f.endswith('.h5') or f.endswith('.keras'))],
            'gender_model': [f for f in current_dir_files if f.startswith('gender') and (f.endswith('.h5') or f.endswith('.keras'))]
        }
        
        if all([len(files) > 0 for files in model_files.values()]):
            # ใช้โมเดลจากไดเร็กทอรีปัจจุบันแทน
            face_model_path = model_files['face_model'][0]
            face_proto_path = model_files['face_proto'][0]
            age_model_path = model_files['age_model'][0]
            gender_model_path = model_files['gender_model'][0]
        else:
            st.warning("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่าได้วางไฟล์โมเดลไว้ในโฟลเดอร์ 'models/' หรือในไดเร็กทอรีปัจจุบัน")
            st.info("คุณต้องมีไฟล์โมเดลดังต่อไปนี้: โมเดลตรวจจับใบหน้า, ไฟล์ config, โมเดลทำนายอายุ และโมเดลทำนายเพศ")
            return None, None, None
    
    try:
        # โหลดโมเดลตรวจจับใบหน้า
        faceNet = cv2.dnn.readNet(face_model_path, face_proto_path)
        
        # โหลดโมเดลทำนายอายุและเพศ
        ageModel = load_model(age_model_path, custom_objects={'lrn': lrn})
        genderModel = load_model(gender_model_path, custom_objects={'lrn': lrn})
        
        return faceNet, ageModel, genderModel
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None, None

# รายการอายุที่ทำนายได้
ageList = ['0-2 years', '4-6 years', '8-12 years', '15-20 years', '25-32 years', '38-43 years', '48-53 years', '60-100 years']
genderList = ['Female', 'Male']

# ฟังก์ชันสำหรับตรวจจับใบหน้า
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # วาดกรอบใบหน้า
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    
    return frameOpencvDnn, faceBoxes

# ฟังก์ชันสำหรับ preprocess ภาพสำหรับโมเดลอายุ
def preprocess_for_age_model(face_img):
    try:
        # ปรับขนาดให้เหมาะกับโมเดลอายุ
        face_resized = cv2.resize(face_img, (224, 224))
        # Normalize ให้อยู่ในช่วง [0, 1]
        face_normalized = face_resized.astype("float") / 255.0
        # เพิ่มมิติแรก (batch dimension)
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    except Exception as e:
        st.error(f"Error in age preprocessing: {e}")
        return None

# ฟังก์ชันสำหรับ preprocess ภาพสำหรับโมเดลเพศ
def preprocess_for_gender_model(face_img):
    try:
        # ปรับขนาดให้เหมาะกับโมเดลเพศ
        face_resized = cv2.resize(face_img, (224, 224))  # CaffeNet ใช้ input size 227x227
        # Normalize ให้อยู่ในช่วง [0, 1]
        face_normalized = face_resized.astype("float") / 255.0
        # เพิ่มมิติแรก (batch dimension)
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    except Exception as e:
        st.error(f"Error in gender preprocessing: {e}")
        return None

# ฟังก์ชันสำหรับการทำนายเพศแบบบังคับให้เป็นแค่ชายหรือหญิง
def predict_binary_gender(gender_preds):
    female_conf = gender_preds[0][0]  # ความมั่นใจที่เป็นผู้หญิง (index 0)
    male_conf = gender_preds[0][1]    # ความมั่นใจที่เป็นผู้ชาย (index 1)
    
    # เลือกเพศที่มีความมั่นใจสูงสุดระหว่างชายและหญิง
    if female_conf > male_conf:
        gender_idx = 0  # Female
        confidence = female_conf * 100
    else:
        gender_idx = 1  # Male
        confidence = male_conf * 100
    
    gender = genderList[gender_idx]
    return gender, confidence

# ฟังก์ชันสำหรับตัดภาพใบหน้าและปรับขนาดให้เท่ากัน
def extract_face_with_fixed_size(frame, face_box, padding=20, target_size=(200, 200)):
    try:
        x1, y1, x2, y2 = face_box
        
        # คำนวณจุดกึ่งกลางของใบหน้า
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        # คำนวณขนาดของใบหน้า (เลือกด้านที่ยาวกว่าเพื่อให้ครอบคลุม)
        face_size = max(x2 - x1, y2 - y1)
        
        # เพิ่ม padding
        face_size_with_padding = face_size + 2 * padding
        
        # คำนวณพิกัดใหม่ที่เป็นสี่เหลี่ยมจัตุรัส
        new_x1 = max(0, face_center_x - face_size_with_padding // 2)
        new_y1 = max(0, face_center_y - face_size_with_padding // 2)
        new_x2 = min(frame.shape[1] - 1, face_center_x + face_size_with_padding // 2)
        new_y2 = min(frame.shape[0] - 1, face_center_y + face_size_with_padding // 2)
        
        # ตัดภาพใบหน้า
        face = frame[new_y1:new_y2, new_x1:new_x2]
        
        # ตรวจสอบว่าภาพถูกต้องหรือไม่
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return None, None
            
        # ปรับขนาดให้เท่ากับ target_size
        face_resized = cv2.resize(face, target_size)
        
        return face_resized, (new_x1, new_y1, new_x2, new_y2)
        
    except Exception as e:
        st.error(f"Error extracting face: {e}")
        return None, None

# ฟังก์ชันสำหรับประมวลผลภาพและทำนายอายุและเพศ
def process_image(img, faceNet, ageModel, genderModel, confidence_threshold=0.7):
    # แปลงรูปแบบจาก PIL มาเป็น OpenCV (RGB to BGR)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # ตรวจจับใบหน้า
    resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold=confidence_threshold)
    
    if not faceBoxes:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [], "ไม่พบใบหน้าในภาพ"
    
    # สร้างรายการสำหรับเก็บข้อมูลใบหน้าและผลการทำนาย
    faces_data = []
    
    # ประมวลผลแต่ละใบหน้า
    for i, faceBox in enumerate(faceBoxes):
        # ตัดภาพใบหน้าและปรับขนาด
        standardized_face, new_face_coords = extract_face_with_fixed_size(frame, faceBox, padding=20)
        
        if standardized_face is None:
            continue
        
        # แปลงเป็นรูปแบบที่โมเดลต้องการ
        face_for_age = preprocess_for_age_model(standardized_face)
        face_for_gender = preprocess_for_gender_model(standardized_face)
        
        # ทำนายเพศ (ใช้ genderModel)
        gender_preds = genderModel.predict(face_for_gender, verbose=0) if face_for_gender is not None else None
        if gender_preds is not None:
            gender, gender_confidence = predict_binary_gender(gender_preds)
        else:
            gender = "Unknown"
            gender_confidence = 0
        
        # ทำนายอายุ (ใช้ ageModel)
        age_preds = ageModel.predict(face_for_age, verbose=0) if face_for_age is not None else None
        if age_preds is not None:
            age_idx = np.argmax(age_preds)
            if age_idx < len(ageList):
                age = ageList[age_idx]
                age_confidence = age_preds[0][age_idx] * 100
            else:
                age = "Unknown"
                age_confidence = 0
        else:
            age = "Unknown"
            age_confidence = 0
        
        # เพิ่มข้อความบนภาพผลลัพธ์
        x1, y1, x2, y2 = faceBox
        label = f"{gender} ({gender_confidence:.1f}%), {age} ({age_confidence:.1f}%)"
        cv2.putText(resultImg, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # เก็บข้อมูลใบหน้า
        face_data = {
            "face_img": cv2.cvtColor(standardized_face, cv2.COLOR_BGR2RGB),
            "gender": gender,
            "gender_confidence": gender_confidence,
            "age": age,
            "age_confidence": age_confidence,
            "box": faceBox
        }
        faces_data.append(face_data)
    
    # แปลงรูปแบบกลับเป็น RGB สำหรับแสดงใน Streamlit
    result_rgb = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    
    message = f"พบใบหน้าทั้งหมด {len(faces_data)} ใบหน้า"
    return result_rgb, faces_data, message

# ฟังก์ชันสำหรับตรวจสอบกล้องที่เชื่อมต่ออยู่
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_available_cameras():
    available_cameras = []
    # ตรวจสอบกล้องตั้งแต่ index 0 ถึง 9 (ปรับตามความเหมาะสม)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # อ่านข้อมูลกล้อง (ถ้าเป็นไปได้)
                camera_name = f"Camera #{i}"
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    camera_name += f" ({width}x{height})"
                available_cameras.append((i, camera_name))
            cap.release()
    
    # ถ้าไม่พบกล้องเลย ให้เพิ่มตัวเลือกเริ่มต้น
    if not available_cameras:
        available_cameras = [(0, "Default Camera")]
    
    return available_cameras

# ฟังก์ชันสำหรับประมวลผลภาพจากกล้องแบบเรียลไทม์
def process_webcam_frame(frame, faceNet, ageModel, genderModel, confidence_threshold=0.7):
    # ตรวจสอบว่ามีภาพเข้ามาหรือไม่
    if frame is None:
        return None, []
    
    # แปลงรูปแบบเป็น OpenCV BGR
    img = np.copy(frame)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # ตรวจจับใบหน้า
    resultImg, faceBoxes = highlightFace(faceNet, img, conf_threshold=confidence_threshold)
    
    # ถ้าไม่พบใบหน้า ให้คืนค่าภาพเดิม
    if not faceBoxes:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), []
    
    # สร้างรายการสำหรับเก็บข้อมูลใบหน้าและผลการทำนาย
    faces_data = []
    
    # ประมวลผลแต่ละใบหน้า
    for i, faceBox in enumerate(faceBoxes):
        # ตัดภาพใบหน้าและปรับขนาด
        standardized_face, new_face_coords = extract_face_with_fixed_size(img, faceBox, padding=20)
        
        if standardized_face is None:
            continue
        
        # แปลงเป็นรูปแบบที่โมเดลต้องการ
        face_for_age = preprocess_for_age_model(standardized_face)
        face_for_gender = preprocess_for_gender_model(standardized_face)
        
        # ทำนายเพศ (ใช้ genderModel)
        gender_preds = genderModel.predict(face_for_gender, verbose=0) if face_for_gender is not None else None
        if gender_preds is not None:
            gender, gender_confidence = predict_binary_gender(gender_preds)
        else:
            gender = "Unknown"
            gender_confidence = 0
        
        # ทำนายอายุ (ใช้ ageModel)
        age_preds = ageModel.predict(face_for_age, verbose=0) if face_for_age is not None else None
        if age_preds is not None:
            age_idx = np.argmax(age_preds)
            if age_idx < len(ageList):
                age = ageList[age_idx]
                age_confidence = age_preds[0][age_idx] * 100
            else:
                age = "Unknown"
                age_confidence = 0
        else:
            age = "Unknown"
            age_confidence = 0
        
        # เพิ่มข้อความบนภาพผลลัพธ์
        x1, y1, x2, y2 = faceBox
        label = f"{gender} ({gender_confidence:.1f}%), {age} ({age_confidence:.1f}%)"
        cv2.putText(resultImg, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # เก็บข้อมูลใบหน้า
        face_data = {
            "face_img": cv2.cvtColor(standardized_face, cv2.COLOR_BGR2RGB),
            "gender": gender,
            "gender_confidence": gender_confidence,
            "age": age,
            "age_confidence": age_confidence,
            "box": faceBox
        }
        faces_data.append(face_data)
    
    # แปลงรูปแบบกลับเป็น RGB สำหรับแสดงใน Streamlit
    result_rgb = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
    
    return result_rgb, faces_data

# Callback function สำหรับ video_processor
def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")
    
    # ใช้ Session State เพื่อเก็บค่า confidence_threshold
    confidence_threshold = st.session_state.get('confidence_threshold', 0.7)
    
    # ประมวลผลเฟรม
    result_frame, _ = process_webcam_frame(
        img, 
        st.session_state.faceNet, 
        st.session_state.ageModel, 
        st.session_state.genderModel, 
        confidence_threshold
    )
    
    return av.VideoFrame.from_ndarray(result_frame, format="rgb24")

# สร้าง sidebar สำหรับการตั้งค่า
st.sidebar.title("⚙️ การตั้งค่า")

# ตั้งค่าความมั่นใจในการตรวจจับใบหน้า
confidence_threshold = st.sidebar.slider(
    "ค่าความมั่นใจในการตรวจจับใบหน้า", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7, 
    step=0.05,
    help="ค่าที่สูงจะทำให้ตรวจจับใบหน้าได้แม่นยำขึ้น แต่อาจจะตรวจจับได้น้อยลง"
)

# เก็บค่า confidence_threshold ใน session state
st.session_state['confidence_threshold'] = confidence_threshold

# เลือกโหมดการทำงาน
mode = st.sidebar.radio("เลือกโหมดการทำงาน", ["อัปโหลดรูปภาพ", "ใช้กล้องเว็บแคม"])

# ส่วนข้อมูลเพิ่มเติม
with st.sidebar.expander("ℹ️ เกี่ยวกับแอปพลิเคชัน"):
    st.markdown("""
    แอปพลิเคชันนี้ใช้โมเดล Deep Learning เพื่อ:
    - ตรวจจับใบหน้าด้วย OpenCV DNN
    - วิเคราะห์เพศและช่วงอายุ
    - แสดงผลลัพธ์พร้อมค่าความมั่นใจ
    
    **หมายเหตุ:** ผลการทำนายอาจมีความคลาดเคลื่อนได้
    """)

# โหลดโมเดล
faceNet, ageModel, genderModel = load_models()

# เก็บโมเดลใน session state เพื่อใช้ใน callback function
st.session_state.faceNet = faceNet
st.session_state.ageModel = ageModel
st.session_state.genderModel = genderModel

# ตรวจสอบว่าโมเดลโหลดสำเร็จหรือไม่
if faceNet is None or ageModel is None or genderModel is None:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง")
    
    # แสดงข้อมูลการตั้งค่าโฟลเดอร์
    st.info("กรุณาสร้างโฟลเดอร์ 'models' และวางไฟล์โมเดลต่อไปนี้:")
    st.code("""
    Age_Gender_Detect_photo/
    ├── opencv_face_detector_uint8.pb  # โมเดลตรวจจับใบหน้า
    ├── opencv_face_detector.pbtxt    # ไฟล์ config ของโมเดลตรวจจับใบหน้า
    ├── age_model_improved.h5         # โมเดลทำนายอายุ
    └── gender_model_improved.h5      # โมเดลทำนายเพศ
    """)
    st.stop()

# แสดงตัวคั่นระหว่างส่วนหัวกับส่วนเนื้อหา
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนการรับข้อมูลนำเข้าตามโหมดที่เลือก
if mode == "อัปโหลดรูปภาพ":
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # แสดงสถานะกำลังประมวลผล
        with st.spinner("กำลังประมวลผลรูปภาพ..."):
            # อ่านรูปภาพ
            image = Image.open(uploaded_file)
            
            # แสดงรูปภาพต้นฉบับ
            st.image(image, caption="รูปภาพต้นฉบับ", use_column_width=True)
            
            # ประมวลผลรูปภาพ
            result_img, faces_data, message = process_image(image, faceNet, ageModel, genderModel, confidence_threshold)
            
            # แสดงผลการตรวจจับใบหน้า
            st.markdown(f"### ผลการวิเคราะห์: {message}")
            
            if len(faces_data) > 0:
                # แสดงรูปภาพผลลัพธ์
                st.image(result_img, caption="ผลการตรวจจับใบหน้า", use_container_width=True)
                
                # แสดงรายละเอียดของแต่ละใบหน้า
                st.markdown("### รายละเอียดใบหน้า")
                
                # สร้าง grid สำหรับแสดงรายละเอียดใบหน้า
                st.markdown("<div class='face-grid'>", unsafe_allow_html=True)
                
                # แบ่งเป็นคอลัมน์ตามจำนวนใบหน้า (สูงสุด 3 คอลัมน์)
                cols = st.columns(min(3, len(faces_data)))
                
                # แสดงข้อมูลแต่ละใบหน้า
                for i, face_data in enumerate(faces_data):
                    col_idx = i % min(3, len(faces_data))
                    
                    with cols[col_idx]:
                        st.image(face_data["face_img"], caption=f"ใบหน้าที่ {i+1}", width=200)
                        
                        # สร้างกล่องข้อมูล
                        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                        
                        # แสดงข้อมูลเพศและอายุพร้อมความมั่นใจ
                        st.markdown(f"**เพศ:** {face_data['gender']} ({face_data['gender_confidence']:.1f}%)")
                        st.markdown(f"**อายุ:** {face_data['age']} ({face_data['age_confidence']:.1f}%)")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # สร้างปุ่มดาวน์โหลดรูปภาพผลลัพธ์
                result_img_pil = Image.fromarray(result_img)
                buf = io.BytesIO()
                result_img_pil.save(buf, format="JPEG")
                byte_img = buf.getvalue()
                
                st.download_button(
                    label="📥 ดาวน์โหลดผลลัพธ์",
                    data=byte_img,
                    file_name="result.jpg",
                    mime="image/jpeg",
                )

else:  # โหมดกล้องเว็บแคม
    # ตรวจสอบกล้องที่มีอยู่
    available_cameras = get_available_cameras()
    camera_options = [f"{cam_name}" for _, cam_name in available_cameras]
    
    # แสดงตัวเลือกกล้อง
    st.markdown("<div class='camera-options'>", unsafe_allow_html=True)
    st.subheader("🎥 เลือกกล้อง")
    
    selected_camera_name = st.selectbox(
        "เลือกกล้องที่ต้องการใช้งาน:",
        options=camera_options,
        index=0
    )
    
    # ค้นหา index ของกล้องที่เลือก
    selected_camera_index = 0
    for idx, name in available_cameras:
        if name == selected_camera_name:
            selected_camera_index = idx
            break
    
    # เลือกโหมดการใช้งานกล้อง
    st.markdown("<div class='camera-mode-selector'>", unsafe_allow_html=True)
    st.subheader("📷 เลือกโหมดการใช้งานกล้อง")
    
    camera_mode = st.radio(
        "โหมดการทำงาน:",
        options=["จับภาพก่อนค่อย detect", "detect ในกล้องแบบ real-time"],
        index=0
    )
    
    # แสดงคำอธิบายสำหรับแต่ละโหมด
    if camera_mode == "จับภาพก่อนค่อย detect":
        st.markdown("<p class='mode-info'>ถ่ายภาพจากกล้องก่อน แล้วจึงทำการวิเคราะห์ใบหน้า เพศ และอายุ</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='mode-info'>วิเคราะห์ใบหน้า เพศ และอายุจากกล้องแบบเรียลไทม์</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # เก็บ camera index และ camera mode ใน session state
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = selected_camera_index
    
    if 'camera_mode' not in st.session_state:
        st.session_state.camera_mode = camera_mode
    
    # อัปเดต camera index เมื่อมีการเปลี่ยนแปลง
    if st.session_state.camera_index != selected_camera_index:
        st.session_state.camera_index = selected_camera_index
        # รีเซ็ตสถานะการถ่ายภาพเมื่อเปลี่ยนกล้อง
        if 'camera_image' in st.session_state:
            st.session_state.pop('camera_image')
    
    # อัปเดต camera mode เมื่อมีการเปลี่ยนแปลง
    if st.session_state.camera_mode != camera_mode:
        st.session_state.camera_mode = camera_mode
        # รีเซ็ตสถานะการถ่ายภาพเมื่อเปลี่ยนโหมด
        if 'camera_image' in st.session_state:
            st.session_state.pop('camera_image')
    
    # แสดงกล้องตามโหมดที่เลือก
    if camera_mode == "จับภาพก่อนค่อย detect":
        # สร้างตัวเลือกสำหรับการจับภาพจากกล้อง
        camera_col1, camera_col2 = st.columns(2)
        
        with camera_col1:
            start_camera = st.button("📸 เปิดกล้องถ่ายภาพ", key="start_camera")
        
        with camera_col2:
            process_button = st.button("🔍 วิเคราะห์ภาพ", key="process_image")
            
        # เริ่มการใช้งานกล้อง
        if start_camera or ('camera_active' in st.session_state and st.session_state.camera_active):
            st.session_state.camera_active = True
            
            # ใช้ st.camera_input โดยระบุ camera_index
            try:
                camera_image = st.camera_input(
                    "กล้องเว็บแคม",
                    key=f"camera_{selected_camera_index}",
                    disabled=False,
                    help="ถ่ายภาพเพื่อวิเคราะห์ใบหน้า"
                )
                
                # เก็บภาพที่ถ่ายไว้ใน session state
                if camera_image is not None:
                    st.session_state.camera_image = camera_image
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเปิดกล้อง: {e}")
                st.info("อาจเกิดจากไม่มีกล้องเว็บแคมหรือกล้องถูกใช้งานโดยโปรแกรมอื่นอยู่")
        
        # ถ้ามีภาพที่ถ่ายไว้ หรือกดปุ่มประมวลผล
        if process_button and 'camera_image' in st.session_state:
            camera_image = st.session_state.camera_image
            
            # แสดงสถานะกำลังประมวลผล
            with st.spinner("กำลังประมวลผลรูปภาพ..."):
                try:
                    # อ่านรูปภาพ
                    image = Image.open(camera_image)
                    
                    # ประมวลผลรูปภาพ
                    result_img, faces_data, message = process_image(image, faceNet, ageModel, genderModel, confidence_threshold)
                    
                    # แสดงผลการตรวจจับใบหน้า
                    st.markdown(f"### ผลการวิเคราะห์: {message}")
                    
                    if len(faces_data) > 0:
                        # แสดงรูปภาพผลลัพธ์
                        st.image(result_img, caption="ผลการตรวจจับใบหน้า", use_column_width=True)
                        
                        # แสดงรายละเอียดของแต่ละใบหน้า
                        st.markdown("### รายละเอียดใบหน้า")
                        
                        # สร้าง grid สำหรับแสดงรายละเอียดใบหน้า
                        cols = st.columns(min(3, len(faces_data)))
                        
                        # แสดงข้อมูลแต่ละใบหน้า
                        for i, face_data in enumerate(faces_data):
                            col_idx = i % min(3, len(faces_data))
                            
                            with cols[col_idx]:
                                st.image(face_data["face_img"], caption=f"ใบหน้าที่ {i+1}", width=200)
                                
                                # สร้างกล่องข้อมูล
                                st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                                
                                # แสดงข้อมูลเพศและอายุพร้อมความมั่นใจ
                                st.markdown(f"**เพศ:** {face_data['gender']} ({face_data['gender_confidence']:.1f}%)")
                                st.markdown(f"**อายุ:** {face_data['age']} ({face_data['age_confidence']:.1f}%)")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                        # สร้างปุ่มดาวน์โหลดรูปภาพผลลัพธ์
                        result_img_pil = Image.fromarray(result_img)
                        buf = io.BytesIO()
                        result_img_pil.save(buf, format="JPEG")
                        byte_img = buf.getvalue()
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์",
                            data=byte_img,
                            file_name="result.jpg",
                            mime="image/jpeg",
                        )
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")
                    st.info("ลองถ่ายภาพใหม่อีกครั้ง")
        
        # ปุ่มรีเซ็ตกล้อง
        if 'camera_active' in st.session_state and st.session_state.camera_active:
            if st.button("🔄 รีเซ็ตกล้อง", key="reset_camera"):
                # รีเซ็ตสถานะกล้อง
                if 'camera_image' in st.session_state:
                    st.session_state.pop('camera_image')
                st.session_state.camera_active = False
                st.experimental_rerun()
    
    # ส่วนของโค้ดสำหรับโหมด real-time ที่ปรับปรุงใหม่
    else:  # โหมด detect ในกล้องแบบ real-time
        # สร้าง placeholder สำหรับแสดงสถานะและผลลัพธ์
        status_placeholder = st.empty()
        frame_placeholder = st.empty()  # placeholder สำหรับแสดงภาพจากกล้อง
        result_placeholder = st.empty()  # placeholder สำหรับแสดงข้อมูลการวิเคราะห์
        
        # ปุ่มควบคุมการทำงาน
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ เริ่มการตรวจจับแบบเรียลไทม์", type="primary")
        with col2:
            stop_button = st.button("⏹️ หยุดการตรวจจับ", type="secondary")
        
        # ตั้งค่าสถานะการทำงาน
        if start_button:
            st.session_state.realtime_active = True
        if stop_button:
            st.session_state.realtime_active = False
            
        # ตรวจสอบสถานะการทำงาน
        if not 'realtime_active' in st.session_state:
            st.session_state.realtime_active = False
            
        if st.session_state.realtime_active:
            status_placeholder.info("กำลังเปิดกล้องเพื่อตรวจจับใบหน้าแบบเรียลไทม์...")
            
            try:
                # ตั้งค่ากล้อง
                frameWidth = 640
                frameHeight = 480
                brightness = 180
                
                # เปิดการเชื่อมต่อกับกล้อง
                cap = cv2.VideoCapture(selected_camera_index)
                cap.set(3, frameWidth)
                cap.set(4, frameHeight)
                cap.set(10, brightness)
                
                # ตรวจสอบว่าเปิดกล้องได้หรือไม่
                if not cap.isOpened():
                    status_placeholder.error("ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง")
                    st.session_state.realtime_active = False
                else:
                    # ทำงานไปเรื่อยๆ จนกว่าจะหยุด
                    while st.session_state.realtime_active:
                        success, img = cap.read()
                        if not success:
                            status_placeholder.error("ไม่สามารถรับภาพจากกล้องได้")
                            break
                        
                        # แปลงสีจาก BGR เป็น RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # ประมวลผลภาพ
                        result_img, faces_data, message = process_image(
                            Image.fromarray(img_rgb), 
                            faceNet, 
                            ageModel, 
                            genderModel, 
                            confidence_threshold
                        )
                        
                        # อัพเดตสถานะ
                        status_placeholder.success(f"กำลังตรวจจับใบหน้าแบบเรียลไทม์... พบ {len(faces_data)} ใบหน้า")
                        
                        # แสดงผลบนหน้าเว็บ (อัพเดตทับตำแหน่งเดิม)
                        frame_placeholder.image(result_img, caption="การตรวจจับแบบเรียลไทม์", use_column_width=True)
                        
                        # แสดงผลข้อมูลใบหน้า (อัพเดตทับตำแหน่งเดิม)
                        # ส่วนแสดงผลข้อมูลใบหน้า (อัพเดตทับตำแหน่งเดิม)
                        # Replace the existing block:
                        if len(faces_data) > 0:
                            st.markdown("<h3 style='color: white; text-align: left;'>👤 รายละเอียดใบหน้า</h3>", unsafe_allow_html=True)
                            
                            # Delete the entire `face_html` generation block
                            
                            # Instead, you can add a simple text display if you want any output
                            for i, face_data in enumerate(faces_data):
                                st.write(f"ใบหน้าที่ {i+1}: {face_data['gender']} ({face_data['gender_confidence']:.1f}%), {face_data['age']} ({face_data['age_confidence']:.1f}%)")
                        else:
                            # Ensure result_placeholder is cleared
                            result_placeholder.empty()
                        
                        # ตรวจสอบการกดปุ่มหยุด
                        if not st.session_state.realtime_active:
                            break
                            
                        # รอเล็กน้อยเพื่อไม่ให้ CPU ทำงานหนักเกินไป
                        time.sleep(0.03)
                    
                # ปิดการใช้งานกล้อง
                cap.release()
                status_placeholder.info("หยุดการตรวจจับแล้ว")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเริ่มการตรวจจับแบบเรียลไทม์: {e}")
                st.session_state.realtime_active = False
        else:
            status_placeholder.info("คลิกที่ปุ่ม 'เริ่มการตรวจจับแบบเรียลไทม์' เพื่อเริ่มใช้งาน")

# แสดงส่วนท้ายเว็บไซต์
st.markdown("<div class='footer'>Face Age Gender Detection App | พัฒนาด้วย Streamlit และ OpenCV</div>", unsafe_allow_html=True)