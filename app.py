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
model_signal = models["yolo26x"]

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
    R, G, B = car_crop_rgb[:,:,0].astype(float), car_crop_rgb[:,:,1].astype(float), car_crop_rgb[:,:,2].astype(float)
    # Flag: If Blue is dominant and significantly brighter than other channels
    blue_mask = (B > R) & (B > G) & (B > 50) & (B > (R + G) * 0.65)
    return np.count_nonzero(blue_mask) / (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])

def get_color_modes(img):
    return {
        "RGB": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "BGR": img.copy(),
        "Grey": cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    }

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

    # --- FLAG 1: VERIFY CLASS INDICES ---
    if debug_mode:
        debug_logs.append(f"Model Classes Found: {list(models['car_expert'].names.values())[:10]}...")

    # --- STEP 1: CAR DETECTION ---
    mid_h, mid_w, margin = h // 2, w // 2, 10
    car_classes = [90, 223, 312, 522] # OIV7 indices for vehicle types
    
    # Quadrant Analysis
    quads = [img_rgb[0:mid_h, 0:mid_w], img_rgb[0:mid_h, mid_w:w],
             img_rgb[mid_h:h, 0:mid_w], img_rgb[mid_h:h, mid_w:w]]
    
    q_internal_sum = 0
    for idx, q_img in enumerate(quads):
        res = models["car_expert"].predict(q_img, imgsz=640, conf=0.25, classes=car_classes, verbose=False)[0]
        if debug_mode: debug_logs.append(f"Quadrant {idx+1} found {len(res.boxes)} potential cars.")
        for box in res.boxes.xyxy.cpu().numpy():
            if not (box[0] <= margin or box[2] >= (w//2)-margin or box[1] <= margin or box[3] >= (h//2)-margin):
                q_internal_sum += 1

    # Global Detection
    whole_res = models["car_expert"].predict(img_rgb, imgsz=640, conf=0.25, classes=car_classes, verbose=False)[0]
    saved_cars = []
    for box in whole_res.boxes.xyxy.cpu().numpy():
        if not is_duplicate(box, saved_cars, 0.6):
            saved_cars.append(box)

    final_car_count = max(len(saved_cars), q_internal_sum + sum(1 for b in saved_cars if (b[0] < mid_w < b[2]) or (b[1] < mid_h < b[3])))

    # Color Filtering
    blue_count = 0
    for box in saved_cars:
        x1, y1, x2, y2 = map(int, box)
        blue_ratio = is_blue_car_robust(img_rgb[y1:y2, x1:x2])
        if blue_ratio > 0.30:
            blue_count += 1
            tmp_cars_blue.append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Blue cars tagged Red for visibility
        else:
            tmp_cars_other.append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # --- STEP 2: SIGNAL DETECTION ---
    modes = list(get_color_modes(img_bgr).values())
    for mode in modes:
        res = model_signal.predict(mode, imgsz=1280, conf=0.05, classes=[9], verbose=False)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            if not is_duplicate(box, tmp_signals):
                tmp_signals.append(box.tolist())

    # --- FLAG 2: SIGNAL FALLBACK TRIGGERED? ---
    if not tmp_signals:
        if debug_mode: debug_logs.append("Primary Signal Detection failed. Running strip-scan fallback.")
        h_steps, w_steps = np.linspace(0, h, 11).astype(int), np.linspace(0, w, 11).astype(int)
        
        for i in range(10): # Horizontal
            y1, y2 = h_steps[i], h_steps[i+1]
            for mode in modes:
                strip = cv2.resize(mode[y1:y2, 0:w], (640, 640))
                res = model_signal.predict(strip, conf=0.05, classes=[9], verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    g_box = [box[0]*(w/640), box[1]*((y2-y1)/640)+y1, box[2]*(w/640), box[3]*((y2-y1)/640)+y1]
                    if not is_duplicate(g_box, tmp_signals): tmp_signals.append(g_box)
        
        for j in range(10): # Vertical
            x1, x2 = w_steps[j], w_steps[j+1]
            strip_w = x2 - x1
            for mode in modes:
                strip = cv2.resize(mode[0:h, x1:x2], (640, 640))
                res = model_signal.predict(strip, conf=0.05, classes=[9], verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    g_box = [box[0]*(strip_w/640)+x1, box[1]*(h/640), box[2]*(strip_w/640)+x1, box[3]*(h/640)]
                    if not is_duplicate(g_box, tmp_signals): tmp_signals.append(g_box)

    # --- STEP 3: PEOPLE ENSEMBLE ---
    p_count, scene = 0, "Normal Scene"
    if tmp_signals:
        scene = "Traffic Signal Scene"
        all_p, all_c = [], []
        for name in ["yolo26n", "yolo26s", "yolo26x", "idd_v8", "lvis_v8"]:
            for img_data in modes:
                res = models[name].predict(img_data, imgsz=1280, conf=0.30, classes=[0], verbose=False)[0]
                for box in res.boxes:
                    all_p.append(box.xyxy[0].cpu().numpy().tolist())
                    all_c.append(float(box.conf[0]))
        
        idxs = cv2.dnn.NMSBoxes(all_p, all_c, 0.30, 0.85)
        if len(idxs) > 0:
            for i in idxs.flatten():
                box = all_p[i]
                tmp_people.append(box)
                cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            p_count = len(idxs)

    for b in tmp_signals:
        cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 3)

    return display_img, final_car_count, blue_count, scene, p_count, debug_logs

# --- 4. STREAMLIT GUI ---
st.set_page_config(page_title="Traffic Intelligence", layout="wide")
st.title("Traffic Scene Intelligence")

# Debug Flag Sidebar
with st.sidebar:
    st.header("Debug Controls")
    debug_active = st.checkbox("Enable Debug Mode", value=False)
    st.write("---")
    st.write("**Model Status:**")
    for m_name in models.keys():
        st.write(f"✅ {m_name} loaded")

uploaded_file = st.file_uploader("Upload image for analysis", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Process
    res_img, t_cars, b_cars, scene, p_counts, logs = process_image(uploaded_file, debug_active)
    
    # Display Result
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Processed Scene")
    
    with col2:
        st.subheader("Inference Results")
        st.metric("Total Cars", t_cars)
        st.metric("Blue Cars", b_cars)
        st.metric("People Count", p_counts)
        st.info(f"Detected Scene: **{scene}**")
        
        if debug_active:
            st.warning("Debug Logs:")
            for log in logs:
                st.write(f"- {log}")
