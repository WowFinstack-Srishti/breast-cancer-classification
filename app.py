import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import sys

# --- 1. IMPORT FROM YOUR PROJECT ---
# Add project root to path to ensure we can import from 'src' and 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.models import ResNet50Fine
    from src.xai import XAIProcessor, overlay_heatmap
    from scripts.stain_norm import normalize_staining # <-- CRITICAL IMPORT
except ImportError:
    print("="*50)
    print("ERROR: Could not import from 'src' or 'scripts' folder.")
    print("Please make sure 'app.py' is in the SAME folder")
    print("as your 'src' and 'scripts' folders.")
    print("="*50)
    exit()

# --- 2. SETUP MODEL AND CONSTANTS ---

# ⚙️ --- CONFIGURE THIS --- ⚙️
MODEL_PATH = 'experiments/federated_run_final/best_resnet.pt'
# ⚙️ --------------------- ⚙️

CLASS_NAMES = {0: 'Normal', 1: 'Benign', 2: 'In-situ', 3: 'Invasive'}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"--- [Gradio App] Using device: {device} ---")

# --- 3. LOAD THE TRAINED MODEL ---
def load_trained_model(model_path):
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at: {model_path}")
        return None
    print(f"Loading model from {model_path}...")
    model = ResNet50Fine(num_classes=4)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
    return model

model = load_trained_model(MODEL_PATH)
if model:
    xai_processor = XAIProcessor(model, device, model_type='resnet')

# --- 4. DEFINE IMAGE TRANSFORMS ---
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. CREATE THE PREDICTION FUNCTION (MODIFIED) ---
def classify_and_explain(input_image_pil):
    if model is None:
        return "ERROR: Model is not loaded.", None, None
        
    # 'input_image_pil' comes from Gradio as a PIL Image
    
    # 1. --- NEW: APPLY STAIN NORMALIZATION ---
    print("Applying stain normalization...")
    # Convert PIL image to NumPy array for the function
    input_image_np = np.array(input_image_pil)
    # Apply the stain normalization
    normalized_image_np = normalize_staining(input_image_np)
    # Convert the result back to a PIL Image for further processing
    normalized_image_pil = Image.fromarray(normalized_image_np)
    
    # 2. Preprocess the *normalized* image
    img_tensor = preprocess(normalized_image_pil).unsqueeze(0).to(device)

    # 3. Get prediction
    with torch.no_grad():
        output_logits = model(img_tensor)
        
    # 4. Get probabilities and predicted class
    probabilities = torch.nn.functional.softmax(output_logits, dim=1)[0]
    pred_idx = probabilities.argmax().item()
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx].item()

    text_output = (
        f"Prediction: {pred_label}\n"
        f"Confidence: {confidence:.2%}"
    )

    # 5. Generate XAI Explanation (Grad-CAM) using the *normalized* image
    print(f"Generating Grad-CAM for class: {pred_label} ({pred_idx})")
    heatmap = xai_processor.gradcam(normalized_image_pil, target_class=pred_idx)
    
    # 6. Overlay heatmap on the *normalized* image
    overlay = overlay_heatmap(normalized_image_pil.resize((224, 224)), heatmap)

    # Return all three outputs for the UI
    return text_output, normalized_image_pil, overlay

# --- 6. LAUNCH THE GRADIO INTERFACE (MODIFIED) ---
print("Launching Gradio UI...")

title = "🩺 Federated AI for Breast Cancer Histopathology"
description = """
Upload a histopathology patch to classify it.
**Step 1:** The image first undergoes stain normalization to ensure color consistency.
**Step 2:** The normalized image is fed into a ResNet50 model trained with Federated Learning (FL).
**Step 3:** The model's classification and a Grad-CAM explanation are displayed.
"""

# Create a more advanced UI with Tabs
with gr.Blocks() as iface:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Histopathology Image")
            submit_btn = gr.Button("Classify Image", variant="primary")
        with gr.Column():
            output_textbox = gr.Textbox(label="Result")
            output_explanation = gr.Image(type="pil", label="Explanation (Grad-CAM)")
            
    with gr.Accordion("Show Preprocessing Details", open=False):
        with gr.Row():
            # Use the original uploaded image for the 'Original' view
            original_view = gr.Image(type="pil", label="Original Image")
            # This will hold the result of our stain normalization
            normalized_view = gr.Image(type="pil", label="Stain Normalized Image")

    # Connect the button click to the function
    submit_btn.click(
        fn=classify_and_explain,
        inputs=input_image,
        outputs=[output_textbox, normalized_view, output_explanation]
    )
    
    # Also link the input image to the 'Original' view so it updates on upload
    input_image.change(lambda x: x, inputs=input_image, outputs=original_view)


# Launch the app!
iface.launch()