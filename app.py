from flask import Flask, request, jsonify, render_template
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the Hugging Face model and processor
model_name = "MoonDream/moondream-captioning"  # Replace with the actual model name
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
  
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Preprocess the image and generate the caption
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"description": caption})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
