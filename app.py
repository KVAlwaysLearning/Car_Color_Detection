import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import gdown
import os

# --- 1. INITIALIZE ASSETS ---
@st.cache_resource
def load_all_assets():
    files = {
        "yolo26n.pt": "1ZGTbc_oHmu42n1EE-cEa0TVBtL7zZ-g2",
        "yolo26s.pt": "1FjrI1avV-uC77iFtk41anBJStyXDLp8p",
        "yolo26x.pt": "1Kjlokvxc4IAIXP5c3jh57tVKhQmcZrzl",
        "idd_yolov8.pt": "1OjHEdbX2bPda9UMtVAVxy7axRFsR_oS6",
        "yolov8x-worldv2.pt": "1uxcdOFg08qqtY7IM-GdqG_DdL2O4-KUN",
        "yolov8x-oiv7.pt": "1pZNZfN-iRcV6040OIGmQSSrAMT_5KoM6"
    }

    for filename, drive_id in files.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={drive_id}'
            gdown.download(url, filename, quiet=False)

    # Initialize models
    models = {
        "yolo26n": YOLO('yolo26n.pt'),
        "yolo26s": YOLO('yolo26s.pt'),
        "yolo26x": YOLO('yolo26x.pt'),
        "idd_v8": YOLO('idd_yolov8.pt'),
        "lvis_v8": YOLO('yolov8x-worldv2.pt'),
        "car_expert": YOLO('yolov8x-oiv7.pt')
    }
    return models

models = load_all_assets()

# --- 2. HELPER FUNCTIONS ---
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_duplicate(new_box, saved_boxes, iou_thresh=0.4):
    for saved in saved_boxes:
        if calculate_iou(new_box, saved) > iou_thresh: return True
    return False

def is_blue_car_robust(car_crop_rgb):
    if car_crop_rgb.size == 0: return 0
    # Convert to float for calculation
    R, G, B = car_crop_rgb[:,:,0].astype(float), car_crop_rgb[:,:,1].astype(float), car_crop_rgb[:,:,2].astype(float)
    # Mask for pixels where blue is clearly the dominant color
    blue_mask = (B > R) & (B > G) & (B > 60) & (B > (R + G) * 0.60)
    return np.count_nonzero(blue_mask) / (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])

# --- 3. CORE PROCESSING ---
def process_image(uploaded_file, debug_mode=False):
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None: return None, 0, 0, "Error", 0

    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display_img = img_bgr.copy()
    
    debug_logs = []
    tmp_signals, tmp_cars_blue, tmp_cars_other, tmp_people = [], [], [], []

    # --- STEP 1: CAR DETECTION (OIV7 Model) ---
    # Indices for Car, Van, Truck, Bus in OIV7
    car_classes = [90, 223, 312, 522] 
    
    # Global Detection
    whole_res = models["car_expert"].predict(img_rgb, imgsz=640, conf=0.25, verbose=False)[0] # changed car_classes
    
    saved_cars = []
    for box in whole_res.boxes.xyxy.cpu().numpy():
        if not is_duplicate(box, saved_cars, 0.5):
            saved_cars.append(box)

    # Logic for counting and color filtering
    blue_count = 0
    for box in saved_cars:
        x1, y1, x2, y2 = map(int, box)
        blue_ratio = is_blue_car_robust(img_rgb[y1:y2, x1:x2])
        if blue_ratio > 0.25: # Relaxed threshold for better recall
            blue_count += 1
            tmp_cars_blue.append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 100, 0), 3) # Highlight blue cars
        else:
            tmp_cars_other.append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 165, 255), 2) # Other cars

    # --- STEP 2: SIGNAL DETECTION (Using yolo26x) ---
    res_sig = models["yolo26x"].predict(img_rgb, imgsz=1280, conf=0.15, classes=[9], verbose=False)[0]
    for box in res_sig.boxes.xyxy.cpu().numpy():
        tmp_signals.append(box.tolist())
        cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)

    # --- STEP 3: PEOPLE ENSEMBLE ---
    scene = "Traffic Signal Scene" if tmp_signals else "Normal Scene"
    p_count = 0
    
    if tmp_signals:
        # Detect people only if it's a signal scene
        res_p = models["yolo26s"].predict(img_rgb, imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
        for box in res_p.boxes.xyxy.cpu().numpy():
            p_count += 1
            cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    if debug_mode:
        debug_logs.append(f"Image Resolution: {w}x{h}")
        # New Log: Show every class ID detected, even if it's not a car
        detected_ids = whole_res.boxes.cls.cpu().numpy().tolist()
        debug_logs.append(f"Detected Class IDs: {list(set(detected_ids))}") 
        debug_logs.append(f"Raw Detections: {len(whole_res.boxes)}")

    return display_img, len(saved_cars), blue_count, scene, p_count, debug_logs

# --- 4. STREAMLIT GUI ---
st.set_page_config(page_title="Traffic Analysis", layout="wide")
st.title("🚦 Traffic Scene Intelligence")

with st.sidebar:
    st.header("Settings")
    debug_active = st.checkbox("Show Debug Info", value=True)
    st.write("---")
    st.write("**Loaded Models:**")
    for m in models.keys():
        st.write(f"✅ {m}")

uploaded_file = st.file_uploader("Choose a traffic image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner('Analyzing scene...'):
        res_img, t_cars, b_cars, scene, p_counts, logs = process_image(uploaded_file, debug_active)
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Note: use_container_width is the modern replacement for use_column_width
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Analysis Summary")
        st.metric("Total Vehicles", t_cars)
        st.metric("Blue Vehicles", b_cars)
        st.metric("Pedestrians", p_counts)
        st.info(f"Scene Context: **{scene}**")
        
        if debug_active:
            st.divider()
            st.write("**Debug Logs:**")
            for log in logs:
                st.write(f"• {log}")
