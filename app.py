import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import gdown
import os
import streamlit as st
# 1. INITIALIZE ALL MODELS (Cached for Streamlit performance)
@st.cache_resource
def load_all_assets():
    # 1. Define the files and IDs
    files = {
        "yolo26n.pt": "1ZGTbc_oHmu42n1EE-cEa0TVBtL7zZ-g2",
        "yolo26s.pt": "1FjrI1avV-uC77iFtk41anBJStyXDLp8p",
        "yolo26x.pt": "1Kjlokvxc4IAIXP5c3jh57tVKhQmcZrzl",
        "idd_yolov8.pt": "1OjHEdbX2bPda9UMtVAVxy7axRFsR_oS6",
        "yolov8x-worldv2.pt": "1uxcdOFg08qqtY7IM-GdqG_DdL2O4-KUN",
        "yolov8x-oiv7.pt": "1pZNZfN-iRcV6040OIGmQSSrAMT_5KoM6"
    }

    # 2. Download missing files
    for filename, drive_id in files.items():
        if not os.path.exists(filename):
            # Note: st.spinner won't work inside cache_resource easily, 
            # so we use a simple print or st.info
            url = f'https://drive.google.com/uc?id={drive_id}'
            gdown.download(url, filename, quiet=False)

    # 3. Load the models into memory
    models = {
        "yolo26n": YOLO('yolo26n.pt'),
        "yolo26s": YOLO('yolo26s.pt'),
        "yolo26x": YOLO('yolo26x.pt'),
        "idd_v8": YOLO('idd_yolov8.pt'),
        "lvis_v8": YOLO('yolov8x-worldv2.pt'),
        "car_expert": YOLO('yolov8x-oiv7.pt')
    }
    return models
    
# Initialize everything
models = load_all_assets()
# Separate the signal model if needed, or just reference from the dict
model_signal = models["yolo26x"]

# --- COORDINATE STORAGE ---
if 'coords' not in st.session_state:
    st.session_state.coords = {
        "blue_cars": [], "other_cars": [], 
        "signals": [], "people": []
    }

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
    if not saved_boxes: return False
    for saved in saved_boxes:
        if calculate_iou(new_box, saved) > iou_thresh: return True
    return False

def is_blue_car_robust(car_crop_rgb):
    if car_crop_rgb.size == 0: return 0
    R, G, B = car_crop_rgb[:,:,0].astype(float), car_crop_rgb[:,:,1].astype(float), car_crop_rgb[:,:,2].astype(float)
    blue_mask = (B > R) & (B > G) & (B > 50) & (B > (R + G) * 0.65)
    return np.count_nonzero(blue_mask) / (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])

def get_color_modes(img):
    return {
        "RGB": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "BGR": img.copy(),
        "Grey": cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    }

def process_image(uploaded_file):
    # Convert Streamlit buffer to OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display_img = img_bgr.copy()
    
    # Reset coordinates
    st.session_state.coords = {"blue_cars":[], "other_cars":[], "signals":[], "people":[]}

    # --- STEP 1: CAR DETECTION ---
    mid_h, mid_w, margin = h // 2, w // 2, 10
    car_classes = [90, 223, 312, 522]
    
    quads = {"Q1": img_rgb[0:mid_h, 0:mid_w], "Q2": img_rgb[0:mid_h, mid_w:w],
             "Q3": img_rgb[mid_h:h, 0:mid_w], "Q4": img_rgb[mid_h:h, mid_w:w]}
    
    q_internal_sum = 0
    for q_img in quads.values():
        res = models["car_expert"].predict(q_img, imgsz=640, conf=0.25, classes=car_classes, verbose=False)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            if not (box[0] <= margin or box[2] >= (w//2)-margin or box[1] <= margin or box[3] >= (h//2)-margin):
                q_internal_sum += 1

    whole_res = models["car_expert"].predict(img_rgb, imgsz=640, conf=0.25, classes=car_classes, verbose=False)[0]
    saved_cars = []
    for box in whole_res.boxes.xyxy.cpu().numpy():
        if ((box[2]-box[0])*(box[3]-box[1])) < (h * w * 0.98) and not is_duplicate(box, saved_cars, 0.6):
            saved_cars.append(box)

    final_car_count = max(len(saved_cars), q_internal_sum + sum(1 for b in saved_cars if (b[0] < mid_w < b[2]) or (b[1] < mid_h < b[3])))

    blue_count = 0
    for box in saved_cars:
        x1, y1, x2, y2 = map(int, box)
        if is_blue_car_robust(img_rgb[y1:y2, x1:x2]) > 0.30:
            blue_count += 1
            st.session_state.coords["blue_cars"].append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            st.session_state.coords["other_cars"].append(box.tolist())
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # --- STEP 2: SIGNAL DETECTION (Tiered) ---
    color_modes_dict = get_color_modes(img_bgr)
    modes = list(color_modes_dict.values())
    
    for mode in modes:
        res = model_signal.predict(mode, imgsz=1280, conf=0.05, classes=[9], verbose=False)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            if not is_duplicate(box, st.session_state.coords["signals"]):
                st.session_state.coords["signals"].append(box.tolist())

    if not st.session_state.coords["signals"]:
        h_steps, w_steps = np.linspace(0, h, 11).astype(int), np.linspace(0, w, 11).astype(int)
        # Horizontal Strips[cite: 1]
        for i in range(10):
            y1, y2 = h_steps[i], h_steps[i+1]
            for mode in modes:
                strip = cv2.resize(mode[y1:y2, 0:w], (640, 640))
                res = model_signal.predict(strip, conf=0.05, classes=[9], verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    g_box = [box[0]*(w/640), box[1]*((y2-y1)/640)+y1, box[2]*(w/640), box[3]*((y2-y1)/640)+y1]
                    if not is_duplicate(g_box, st.session_state.coords["signals"]): st.session_state.coords["signals"].append(g_box)
        
        # Vertical Strips[cite: 1]
        for j in range(10):
            x1, x2 = w_steps[j], w_steps[j+1]
            for mode in modes:
                strip = cv2.resize(mode[0:h, x1:x2], (640, 640))
                res = model_signal.predict(strip, conf=0.05, classes=[9], verbose=False)[0]
                for box in res.boxes.xyxy.cpu().numpy():
                    g_box = [box[0]*((x2-x1)/640)+x1, box[1]*(h/640), box[2]*((x2-x1)/640)+x1, box[3]*(h/640)]
                    if not is_duplicate(g_box, st.session_state.coords["signals"]): st.session_state.coords["signals"].append(g_box)

    # --- STEP 3: PEOPLE ENSEMBLE ---
    p_count, scene = 0, "Normal Scene"
    if st.session_state.coords["signals"]:
        scene = "Traffic Signal Scene"
        all_p, all_c = [], []

        # Define which models look for people
        people_models = ["yolo26n", "yolo26s", "yolo26x", "idd_v8", "lvis_v8"]
       
        for name in people_models:
            m = models[name]
            for img_data in modes:
                # We filter for class 0 (person) directly in the prediction call
                res = m.predict(img_data, imgsz=1280, conf=0.30, classes=[0], verbose=False)[0]
                
                for box in res.boxes:
                  
                        all_p.append(box.xyxy[0].cpu().numpy().tolist())
                        all_c.append(float(box.conf[0]))
        
        idxs = cv2.dnn.NMSBoxes(all_p, all_c, 0.30, 0.85)
        if len(idxs) > 0:
            for i in idxs.flatten():
                box = all_p[i]
                st.session_state.coords["people"].append(box)
                cv2.rectangle(display_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            p_count = len(idxs)

    for b in st.session_state.coords["signals"]:
        cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 3)

    return display_img, final_car_count, blue_count, scene, p_count

# --- STREAMLIT GUI ---
st.title("Traffic Scene Intelligence")
uploaded_file = st.file_uploader("Upload image for analysis", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    res_img, t_cars, b_cars, scene, p_counts = process_image(uploaded_file)
    st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    st.markdown(f"""
    **Total Cars:** {t_cars}  
    **Blue Cars Count:** {b_cars}  
    **Other Cars Count:** {t_cars - b_cars}  
    **Scene Type:** {scene}  
    **People Counts:** {p_counts}
    """)
