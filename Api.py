from flask import Flask, request, jsonify, send_file
import os
from ultralytics import YOLO

app = Flask(__name__)

HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, "Weights", "best.pt")
OUTPUT_DIR = os.path.join(HOME, "Output")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)



@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(OUTPUT_DIR, file.filename)
    file.save(image_path)


    # Run inference
    results = model(image_path)


    # Save the output image with bounding boxes
    output_image_path = os.path.join(OUTPUT_DIR, "output.jpg")
    results[0].save(output_image_path)

    return send_file(output_image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)