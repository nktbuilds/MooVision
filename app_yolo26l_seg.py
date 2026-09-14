import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Setup Streamlit Page
st.set_page_config(page_title="MooVision - Segmentation", layout="wide")
st.title("MooVision: Cow Segmentation (YOLO26l-Seg)")

# 2. Load the Segmentation Model
@st.cache_resource
def load_seg_model():
    # Notice the '-seg' in the filename. This tells Ultralytics to fetch 
    # the YOLO26 Nano Segmentation model instead of the standard one.
    return YOLO("yolo26l-seg.pt")

model = load_seg_model()

# 3. Create Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Image Feed")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    frame_window = st.empty() 

with col2:
    st.header("Detection Tally")
    counter_box = st.empty()

# 4. Process Uploaded Image
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Run inference, filtering ONLY for class 19 (Cow)
    results = model(frame, classes=[19], conf=0.15, verbose=False)
    
    # Render the annotated frame
    # Because we loaded a '-seg' model, this will automatically draw 
    # colored transparent masks over the exact shape of the cows!
    annotated_frame = results[0].plot()
    
    # Tally detected objects
    counts = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

    # Convert BGR back to RGB for Streamlit
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb, width="stretch")
    
    # Display detection stats
    stats_text = "\n".join([f"**{k.capitalize()}**: {v}" for k, v in counts.items()])
    counter_box.markdown(stats_text if stats_text else "No cows detected.")
else:
    frame_window.info("Please upload an image to begin.")
    counter_box.markdown("Waiting for image...")