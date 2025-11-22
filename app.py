import os
import io
import time
import base64
import uuid
import PIL.Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Google GenAI imports (new unified API)
from google import genai
from google.genai import types

# --- Configuration & Initialization ---
load_dotenv(".env")

app = Flask(__name__)

# Local directories
LOCAL_IMAGE_DIR = os.path.join("static", "generated_images")
LOCAL_VIDEO_DIR = os.path.join("static", "generated_videos")
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_ID_IMAGE = os.environ.get("IMAGE_GEN_MODEL")  # e.g., gemini-2.5-flash-image
MODEL_ID_VIDEO = os.environ.get("VIDEO_GEN_MODEL")  # e.g., veo-3.1-generate-preview

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Initialize GenAI client
try:
    genai_client = genai.Client(api_key=API_KEY)
    print("GenAI client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize GenAI client: {e}")
    genai_client = None


# --- Main Route ---
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------
#                IMAGE GENERATION (2:3 ASPECT)
# --------------------------------------------------------
@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    if not genai_client:
        return jsonify({"error": "GenAI client not initialized"}), 500

    data = request.get_json()
    if not data or "image_data" not in data or "prompt" not in data:
        return jsonify({"error": "Missing 'image_data' or 'prompt'"}), 400

    base64_image_data = data["image_data"]
    user_prompt = data["prompt"].strip()

    try:
        # Decode incoming base64 image
        image_bytes = base64.b64decode(base64_image_data.split(",", 1)[1])
        input_pil_image = PIL.Image.open(io.BytesIO(image_bytes))

        prompt_text = f"Using the input image as a guide: {user_prompt}"

        # ------------------------
        #  IMAGE GEN WITH 2:3 RATIO
        # ------------------------
        response = genai_client.models.generate_content(
            model=MODEL_ID_IMAGE,
            contents=[prompt_text, input_pil_image],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="2:3"   # 👈 Match HTML ratio
                ),
            ),
        )

        if not response.candidates:
            raise ValueError("Gemini returned no candidates.")

        # Extract image bytes
        generated_image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                generated_image_bytes = part.inline_data.data
                break

        if not generated_image_bytes:
            raise ValueError("No image returned by Gemini.")

        generated_image_base64 = base64.b64encode(generated_image_bytes).decode("utf-8")
        return jsonify({"image_base64": generated_image_base64})

    except Exception as e:
        print(f"Image generation error: {e}")
        return jsonify({"error": f"Failed to generate image: {e}"}), 500


# --------------------------------------------------------
#                      VIDEO GENERATION
# --------------------------------------------------------
@app.route("/generate-video", methods=["POST"])
def generate_video_route():
    if not genai_client:
        return jsonify({"error": "GenAI client not initialized"}), 500

    data = request.get_json()
    if not data or "image_data" not in data or "prompt" not in data:
        return jsonify({"error": "Missing 'image_data' or 'prompt'"}), 400

    base64_image_data = data["image_data"]
    video_prompt = data["prompt"].strip()

    try:
        # Convert base64 → PNG bytes → genai image
        img_bytes = base64.b64decode(base64_image_data.split(",", 1)[1])
        pil_image = PIL.Image.open(io.BytesIO(img_bytes))

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        inline_png_bytes = buf.getvalue()

        # Convert to GenAI image part
        image_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=inline_png_bytes)
        )
        image_for_video = image_part.as_image()

        # ================================
        #     CALL VEO 3.1 PREVIEW
        # ================================
        operation = genai_client.models.generate_videos(
            model=MODEL_ID_VIDEO,
            prompt=video_prompt,
            image=image_for_video,
        )

        # Poll the operation
        print("Polling Veo video generation...")
        while not operation.done:
            time.sleep(10)
            operation = genai_client.operations.get(operation)
            print("...still generating...")

        if not operation.response.generated_videos:
            raise ValueError("Veo returned no videos.")

        # Download video to local /static folder
        video = operation.response.generated_videos[0]
        filename = f"video_{uuid.uuid4()}.mp4"
        save_path = os.path.join(LOCAL_VIDEO_DIR, filename)

        genai_client.files.download(file=video.video)
        video.video.save(save_path)

        print(f"Video saved locally: {save_path}")

        public_url = f"/static/generated_videos/{filename}"

        return jsonify({"generated_video_url": public_url})

    except Exception as e:
        print(f"Video generation error: {e}")
        return jsonify({"error": f"Failed to generate video: {e}"}), 500


# --------------------------------------------------------
#                        RUN FLASK
# --------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)