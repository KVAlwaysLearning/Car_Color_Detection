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

    return {
        "yolo26n": YOLO('yolo26n.pt'),
        "yolo26s": YOLO('yolo26s.pt'),
        "yolo26x": YOLO('yolo26x.pt'),
        "idd_v8": YOLO('idd_yolov8.pt'),
        "lvis_v8": YOLO('yolov8x-worldv2.pt'),
        "car_expert": YOLO('yolov8x-oiv7.pt')
    }

models = load_all_assets()

# --- 2. HELPER FUNCTIONS ---
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1, xi2, yi2 = max(x1_1, x1_2), max(y1_1, y1_2), min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_duplicate(new_box, saved_boxes, iou_thresh=0.6):
    for saved in saved_boxes:
        if calculate_iou(new_box, saved) > iou_thresh: return True
    return False

def is_blue_car_robust(car_crop_rgb):
    if car_crop_rgb.size == 0: return 0
    R, G, B = car_crop_rgb[:,:,0].astype(float), car_crop_rgb[:,:,1].astype(float), car_crop_rgb[:,:,2].astype(float)
    blue_mask = (B > R) & (B > G) & (B > 50) & (B > (R + G) * 0.65)
    return np.count_nonzero(blue_mask) / (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])

# --- 3. CORE PROCESSING ---
def process_image(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None, 0, 0, "Error", 0, []

    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display_img = img_bgr.copy()
    
    # --- UPDATED CLASS AGREEMENT MAP ---[cite: 5]
    # Hardcoded IDs specific to each model's training set
    MODEL_ID_MAP = {
        "yolo26n":    {"person": [0], "car": [2, 7], "signal": [9]},
        "yolo26s":    {"person": [0], "car": [2, 7], "signal": [9]},
        "yolo26x":    {"person": [0], "car": [2, 7], "signal": [9]},
        "idd_v8":     {"person": [0, 1, 2], "car": [4], "signal": [11]},
        "lvis_v8":    {"person": [0, 1, 2], "car": [3], "signal": [10]},
        "car_expert": {"person": [68, 566], "car": [90, 223, 312, 522], "signal": [419]}
    }

    # --- STEP 1: CAR DETECTION (Quadrant + Global) ---
    mid_h, mid_w, margin = h // 2, w // 2, 10
    car_ids = MODEL_ID_MAP["car_expert"]["car"] # Use expert car IDs[cite: 5]
    quads = [img_rgb[0:mid_h, 0:mid_w], img_rgb[0:mid_h, mid_w:w],
             img_rgb[mid_h:h, 0:mid_w], img_rgb[mid_h:h, mid_w:w]]
    
    q_internal_sum = 0
    for q_img in quads:
        res = models["car_expert"].predict(q_img, imgsz=640, conf=0.25, classes=car_ids, verbose=False)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            if not (box[0] <= margin or box[2] >= (w//2)-margin or box[1] <= margin or box[3] >= (h//2)-margin):
                q_internal_sum += 1

    whole_res = models["car_expert"].predict(img_rgb, imgsz=640, conf=0.20, classes=car_ids, verbose=False)[0]
    saved_cars = []
    for box in whole_res.boxes.xyxy.cpu().numpy():
        if ((box[2]-box[0])*(box[3]-box[1])) < (h * w * 0.98) and not is_duplicate(box, saved_cars, 0.6):
            saved_cars.append(box)

    boundary_count = sum(1 for b in saved_cars if (b[0] < mid_w < b[2]) or (b[1] < mid_h < b[3]))
    final_car_count = max(len(saved_cars), q_internal_sum + boundary_count)

    blue_count = 0
    for box in saved_cars:
        x1, y1, x2, y2 = map(int, box)
        if is_blue_car_robust(img_rgb[y1:y2, x1:x2]) > 0.30:
            blue_count += 1
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # --- STEP 2: SIGNAL DETECTION (Multi-Mode) ---
    coords_signals = []
    signal_id = MODEL_ID_MAP["yolo26x"]["signal"] # Use signal IDs for yolo26x[cite: 5]
    for mode in [img_rgb, img_bgr]:
        res_sig = models["yolo26x"].predict(mode, imgsz=1280, conf=0.05, classes=signal_id, verbose=False)[0]
        for box in res_sig.boxes.xyxy.cpu().numpy():
            if not is_duplicate(box, coords_signals): coords_signals.append(box.tolist())

    # --- STEP 3: PEOPLE ENSEMBLE (Agreement via NMS) ---
    p_count = 0
    scene = "Traffic Signal Scene" if coords_signals else "Normal Scene"
    if coords_signals:
        all_p_boxes, all_p_confs = [], []
        # Query each model using its unique hardcoded Person ID[cite: 5]
        for name in ["yolo26n", "yolo26s", "yolo26x", "idd_v8", "lvis_v8"]:
            person_ids = MODEL_ID_MAP[name]["person"]
            res_p = models[name].predict(img_rgb, imgsz=1280, conf=0.30, classes=person_ids, verbose=False)[0]
            for box in res_p.boxes:
                all_p_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                all_p_confs.append(float(box.conf[0]))
        
        p_indices = cv2.dnn.NMSBoxes(all_p_boxes, all_p_confs, 0.30, 0.85)
        if len(p_indices) > 0:
            p_count = len(p_indices.flatten())
            for i in p_indices.flatten():
                b = all_p_boxes[i]
                cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)

    for b in coords_signals:
        cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 3)

    logs = [f"Res: {w}x{h}", f"Global Detections: {len(whole_res.boxes)}", f"Signals: {len(coords_signals)}"]
    return display_img, final_car_count, blue_count, scene, p_count, logs

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Integrated Traffic Intel", layout="wide")
st.title("🚦 Integrated Traffic Intelligence")

uploaded_file = st.file_uploader("Upload Scene", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    res_img, t_cars, b_cars, scene, p_counts, logs = process_image(uploaded_file)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.metric("Total Vehicles", t_cars)
        st.metric("Blue Vehicles", b_cars)
        st.metric("Pedestrians", p_counts)
        st.info(f"Scene: {scene}")
        for log in logs: st.write(f"• {log}")
