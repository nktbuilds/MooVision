import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor

# 1. Setup Streamlit Page
st.set_page_config(page_title="MooVision - YOLO26x-Seg + Behaviour sigLIP2", layout="wide")
st.title("MooVision: Cow Segmentation + Behaviour Classification (YOLO26x-Seg + SigLIP2)")

# 2. Behaviour prompts (display name -> text encoder prompt)
BEHAVIOURS = {
    # 1. Lying: Directly captures curled resting rumps and backs in stalls; excludes standing walkway cows
    "Lying": "a photo of a cow curled up resting asleep in a stall or cubicle",

    # 2. Standing: Captures upright cows walking or standing in the barn aisle/corridor
    "Standing": "a photo of an upright cow standing on its feet in the barn corridor",

    # 3. Eating: Specifically captures cows feeding at the green forage fence (fixes Image 3 completely)
    "Eating / Head-Down": "a photo of a cow feeding on green grass or reaching through the feeding fence to eat",
}
BEHAVIOUR_NAMES = list(BEHAVIOURS)
BEHAVIOUR_PROMPTS = list(BEHAVIOURS.values())

CLIP_MODEL_ID = "google/siglip2-base-patch16-256"


# 3. Load the Segmentation Model (existing behaviour, unchanged)
@st.cache_resource
def load_seg_model():
    return YOLO("yolo26x-seg.pt")


# 4. Load the SigLIP model once AND run the 3 behaviour prompts through the
#    text encoder a single time, caching the resulting [3, dim] matrix in RAM.
#    This is the key trick: per cow we ONLY run the image encoder, then do a
#    cheap matrix multiply against the pre-computed text embeddings.
@st.cache_resource
def load_clip_and_text_embeddings(model_id, prompts):
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # SigLIP was trained with padding="max_length", max_length=64.
    text_inputs = processor(
        text=prompts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_out = model.get_text_features(input_ids=text_inputs["input_ids"])
        # Defensive: get pooler output whether the model returns a dataclass or a tuple
        text_embeds = (
            text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out[1]
        )
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    return model, processor, text_embeds


model = load_seg_model()
clip_model, clip_processor, text_embeds = load_clip_and_text_embeddings(
    CLIP_MODEL_ID, BEHAVIOUR_PROMPTS
)

# 5. Sidebar controls
with st.sidebar:
    st.header("CLIP Behaviour Settings")
    box_padding = st.slider("Box padding (%)", min_value=0, max_value=50, value=15)
    st.caption(f"Model: `{CLIP_MODEL_ID}`")
    st.caption("Text prompts encoded once at startup. Each crop runs the image encoder only.")

# 6. Create Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Image Feed")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    frame_window = st.empty()

with col2:
    st.header("Behaviour Tally")
    counter_box = st.empty()


def pad_box(x1, y1, x2, y2, padding, h, w):
    dx = int((x2 - x1) * padding / 100)
    dy = int((y2 - y1) * padding / 100)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w, x2 + dx),
        min(h, y2 + dy),
    )


def classify_behaviour(pil_crop):
    """Run the image encoder on a single crop and score it against the cached text embeddings."""
    inputs = clip_processor(images=pil_crop, return_tensors="pt")
    with torch.no_grad():
        image_out = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds = (
            image_out.pooler_output if hasattr(image_out, "pooler_output") else image_out[1]
        )
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        logits = image_embeds @ text_embeds.t()
        # temperature scaling kept (sharpens softmax); logit_bias is a no-op under softmax
        logits = logits * clip_model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    scores = probs.tolist()
    ranked = sorted(zip(scores, BEHAVIOUR_NAMES), key=lambda x: -x[0])
    return ranked


# 7. Process Uploaded Image
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # Run inference, filtering ONLY for class 19 (Cow)
    results = model(frame, classes=[19], conf=0.15, verbose=False)

    # Render the annotated frame (boxes + segmentation masks)
    annotated_frame = results[0].plot()

    # Iterate over every detected cow: crop with padding -> behaviour classification
    per_cow = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            px1, py1, px2, py2 = pad_box(
                int(x1), int(y1), int(x2), int(y2), box_padding, h, w
            )
            crop_bgr = frame[py1:py2, px1:px2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            ranked = classify_behaviour(crop_pil)
            best_score, best_label = ranked[0]

            # Draw the predicted behaviour on the annotated frame
            label_text = f"#{i+1} {best_label} ({best_score:.2f})"
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            cv2.putText(
                annotated_frame, label_text, (px1, py1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            per_cow.append(
                {"index": i + 1, "crop": crop_pil, "ranked": ranked}
            )

    # Tally behaviours
    counts = {}
    for cow in per_cow:
        best = cow["ranked"][0][1]
        counts[best] = counts.get(best, 0) + 1

    # Convert BGR back to RGB for Streamlit
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb, width="stretch")

    # Display behaviour stats
    stats_text = "\n".join(
        f"- **{k}**: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
    )
    counter_box.markdown(stats_text if stats_text else "No cows detected.")

    # 8. Per-Cow Behaviour Analysis gallery (multi-row grid)
    if per_cow:
        st.subheader("Per-Cow Behaviour Analysis")
        st.caption("Crops are run through the SigLIP image encoder and matched against "
                   "pre-encoded behaviour text prompts (top-3 closest shown).")
        crops_per_row = st.slider("Crops per row", min_value=2, max_value=6, value=4)
        for i in range(0, len(per_cow), crops_per_row):
            chunk = per_cow[i:i + crops_per_row]
            cols = st.columns(len(chunk))
            for cow, col in zip(chunk, cols):
                with col:
                    best_score, best_label = cow["ranked"][0]
                    st.image(
                        cow["crop"],
                        caption=f"Cow #{cow['index']} — {best_label} ({best_score:.2f})",
                    )
                    with st.expander("Top-3"):
                        st.markdown("\n".join(
                            f"- {label}: {score:.3f}" for score, label in cow["ranked"][:3]
                        ))
else:
    frame_window.info("Please upload an image to begin.")
    counter_box.markdown("Waiting for image...")
