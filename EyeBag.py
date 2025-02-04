import os
import gradio as gr
from ultralytics import YOLO

# Set up paths
HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, "Weights", "best.pt")
OUTPUT_DIR = os.path.join(HOME, "Output")


# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Run Model
def detect_objects(image_path):
    # Run inference
    results = model(image_path)

    # Save the output image with bounding boxes
    output_image_path = os.path.join(OUTPUT_DIR, "output.jpg")
    results[0].save(output_image_path)

    return output_image_path


# Gradio UI
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="filepath"),
    title="YOLO Object Detection",
    description="Upload an image, and the YOLO model will detect objects."
)


# Launch the app
iface.launch(share=True)
