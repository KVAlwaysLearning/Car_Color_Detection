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
def is_duplicate(new_box, saved_boxes, iou_thresh=0.4):
    if not saved_boxes: return False
    nx1, ny1, nx2, ny2 = new_box
    area1 = (nx2 - nx1) * (ny2 - ny1)
    for sx1, sy1, sx2, sy2 in saved_boxes:
        area2 = (sx2 - sx1) * (sy2 - sy1)
        ix1, iy1 = max(nx1, sx1), max(ny1, sy1)
        ix2, iy2 = min(nx2, sx2), min(ny2, sy2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inters = iw * ih
        uni = area1 + area2 - inters
        
        # Check standard IoU
        if uni > 0 and (inters / uni) > iou_thresh: return True
        # NEW: Check if one box is almost entirely inside the other (Nested check)
        if inters / min(area1, area2) > 0.85: return True 
    return False


def is_blue_car_robust(car_crop_rgb):
    if car_crop_rgb.size == 0: return 0
    R, G, B = car_crop_rgb[:,:,0].astype(float), car_crop_rgb[:,:,1].astype(float), car_crop_rgb[:,:,2].astype(float)
    blue_mask = (B > R) & (B > G) & (B > 50) & (B > (R + G) * 0.65)
    return np.count_nonzero(blue_mask) / (car_crop_rgb.shape[0] * car_crop_rgb.shape[1])

def get_color_modes(img_bgr):
    return {
        "RGB": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "BGR": img_bgr.copy(),
        "Grey": cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    }

# --- 3. CORE PROCESSING ---
def process_image(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None, 0, 0, "Error", 0, []

    h, w, _ = img_bgr.shape
    modes_dict = get_color_modes(img_bgr)
    img_rgb = modes_dict["RGB"]
    display_img = img_bgr.copy()

    # --- DYNAMIC ID DISCOVERY ---
    # We define keywords to search for in each model's internal .names dictionary
    target_keywords = {
    "person": ["person", "pedestrian", "rider"],
    "car": ["car", "vehicle"],
    "signal": ["traffic light", "traffic_light", "signal"]
    }

    # Generate a mapping for every model loaded
    model_ids = {}
    for name, model in models.items():
        model_ids[name] = {
            key: [id for id, label in model.names.items() 
                  if any(word in label.lower() for word in words)]
            for key, words in target_keywords.items()
        }

    # --- STEP 1: CAR DETECTION ---
    mid_h, mid_w, margin = h // 2, w // 2, 10
    car_ids = model_ids['car_expert']["car"]
    
    q_internal_sum = 0
    quads = [img_rgb[0:mid_h, 0:mid_w], img_rgb[0:mid_h, mid_w:w],
             img_rgb[mid_h:h, 0:mid_w], img_rgb[mid_h:h, mid_w:w]]
    
    for q_img in quads:
        res = models["car_expert"].predict(q_img, imgsz=640, conf=0.25, classes=car_ids, augment=True, verbose=False)[0]
        for box in res.boxes.xyxy.cpu().numpy():
            if not (box[0] <= margin or box[2] >= (w//2)-margin or box[1] <= margin or box[3] >= (h//2)-margin):
                q_internal_sum += 1

    whole_res = models["car_expert"].predict(img_rgb, imgsz=640, conf=0.25, classes=car_ids, augment=True, verbose=False)[0]
    saved_cars = []
    for box in whole_res.boxes.xyxy.cpu().numpy():
        if ((box[2]-box[0])*(box[3]-box[1])) < (h * w * 0.98) and not is_duplicate(box, saved_cars, 0.6):
            saved_cars.append(box)

    boundary_count = sum(1 for b in saved_cars if (b[0] < mid_w < b[2]) or (b[1] < mid_h < b[3]))
    final_car_count = max(len(saved_cars), q_internal_sum + boundary_count)

    blue_count = 0
    for box in saved_cars:
        x1, y1, x2, y2 = map(int, box)
        if is_blue_car_robust(img_rgb[y1:y2, x1:x2]) > 0.15:
            blue_count += 1
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # --- STEP 2: SIGNAL DETECTION (TIERED SCAN) ---
    unique_signals = []
    current_sig_ids = model_ids["yolo26x"]["signal"]
    color_modes_dict = get_color_modes(img_bgr) # img_bgr is what cv2.imdecode produced
    modes_to_test = [color_modes_dict["RGB"], color_modes_dict["BGR"], color_modes_dict["Grey"]]
    
    # Tier 1: Multi-Mode Pass
    for mode in modes_to_test:
        res_sig = models["yolo26x"].predict(
        mode, 
        imgsz=1280, 
        conf=0.03, 
        classes=current_sig_ids,
        verbose=False
    )[0]
        for box in res_sig.boxes.xyxy.cpu().numpy():
        # Use your original is_duplicate check with iou_thresh=0.3
            if not is_duplicate(box, unique_signals, iou_thresh=0.3):
                unique_signals.append(box.tolist())

    # Tier 2: Deep Strip Scan
    if not unique_signals:
        h_steps = np.linspace(0, h, 11).astype(int)
    
        for i in range(10):
            y1, y2 = h_steps[i], h_steps[i+1]
            for m_name in ["RGB", "BGR", "Grey"]:
                strip = modes_dict[m_name][y1:y2, 0:w]
                sh, sw = strip.shape[:2] # Get actual strip dimensions
                res_h = models["yolo26x"].predict(cv2.resize(strip, (640, 640)), conf=0.05, classes=[9], verbose=False)[0]
                for b in res_h.boxes.xyxy.cpu().numpy():
                    # Correct scaling based on the actual strip size vs the 640 target
                    rx1, ry1 = b[0] * (sw/640), b[1] * (sh/640)
                    rx2, ry2 = b[2] * (sw/640), b[3] * (sh/640)
                    g_box = [rx1, ry1 + y1, rx2, ry2 + y1]
                    if not is_duplicate(g_box, unique_signals): 
                        unique_signals.append(g_box)
                        
    # --- STEP 3: PEOPLE ENSEMBLE ---
    p_count = 0
    scene = "Traffic Signal Scene" if unique_signals else "Normal Scene"
    
    if unique_signals:
      all_p_boxes, all_p_confs = [], []
      # Loop through the ensemble and pull the specific 'person' IDs for EACH model
      for name in ["yolo26n", "yolo26s", "yolo26x", "idd_v8", "lvis_v8"]:
          specific_person_ids = model_ids[name]["person"]
        
          for m_name in ["RGB", "BGR", "Grey"]:
              res_p = models[name].predict(modes_dict[m_name], imgsz=1280, conf=0.30, classes=specific_person_ids, verbose=False)[0]
              for box in res_p.boxes:
                all_p_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                all_p_confs.append(float(box.conf[0]))
            
      p_indices = cv2.dnn.NMSBoxes(all_p_boxes, all_p_confs, 0.30, 0.85)
            
      if len(p_indices) > 0:
        p_count = len(p_indices.flatten())
        for i in p_indices.flatten():
            b = all_p_boxes[i]
            cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
    
    #for b in unique_signals:
    #    cv2.rectangle(display_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 1)
    
    # RESIZE OUTPUT TO 256X256
    final_render = cv2.resize(display_img, (768, 768))
    return final_render, final_car_count, blue_count, scene, p_count, unique_signals

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Integrated Traffic Intel", layout="wide")
st.title("🚦 Traffic Scene Intelligence")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])


if uploaded_file:
    res_img, t_cars, b_cars, scene, p_counts,u_sig = process_image(uploaded_file)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), width=768)
        
    with col2:
        st.metric("Total Cars", t_cars)
        st.metric("Blue Cars", b_cars)
        
        # --- CONDITIONAL PEOPLE DISPLAY ---
        if scene == "Normal Scene":
            st.warning("⚠️ People Detection Not Applicable for Non-Traffic Signal Scene")
        else:
            st.metric("Pedestrians", p_counts)
            
        st.info(f"Scene Type: {scene}")
        # Optional: Hide coordinate list if it clutters the UI
        # st.info(f"{u_sig}")
