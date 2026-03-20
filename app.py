from flask import Flask, render_template, request, jsonify
import base64
import os
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="1Pu7fAKql1EZFXUnySxt"
)

translations = {
    "10_rupee": {"en": "10 Rupees", "ta": "பத்து ரூபாய்"},
    "20_rupee": {"en": "20 Rupees", "ta": "இருபது ரூபாய்"},
    "50_rupee": {"en": "50 Rupees", "ta": "ஐம்பது ரூபாய்"},
    "100_rupee": {"en": "100 Rupees", "ta": "நூறு ரூபாய்"},
    "200_rupee": {"en": "200 Rupees", "ta": "இருநூறு ரூபாய்"},
    "500_rupee": {"en": "500 Rupees", "ta": "ஐநூறு ரூபாய்"},
    "2000_rupee": {"en": "2000 Rupees", "ta": "இரண்டாயிரம் ரூபாய்"}
}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    language = data["language"]

    # Decode base64 image
    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    temp_path = "temp.png"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    # Inference
    result = CLIENT.infer(temp_path, model_id="currency-detection-cgpjn/2")
    predictions = result.get("predictions", [])

    if predictions:
        cls = predictions[0]["class"]

        if cls in translations:
            text = translations[cls][language]
        else:
            text = cls
    else:
        text = "No currency detected" if language == "en" else "நாணயம் கண்டறியப்படவில்லை"

    return jsonify({
        "text": text,
        "lang": "en-US" if language == "en" else "ta-IN"
    })


if __name__ == "__main__":
    app.run(debug=True)