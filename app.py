import os
import io
import time
import base64
import uuid
import PIL.Image
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from dotenv import load_dotenv

# Google GenAI imports (new unified API)
from google import genai
from google.genai import types

import utils

# --- Configuration & Initialization ---
load_dotenv(".env")

app = Flask(__name__)

# Local directories
LOCAL_IMAGE_DIR = os.path.join("static", "generated_images")
LOCAL_VIDEO_DIR = os.path.join("static", "generated_videos")
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

API_KEY = os.environ.get("GOOGLE_API_KEY")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
MODEL_ID_IMAGE = os.environ.get("IMAGE_GEN_MODEL")  # e.g., gemini-2.5-flash-image
MODEL_ID_VIDEO = os.environ.get("VIDEO_GEN_MODEL")  # e.g., veo-3.1-generate-preview

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Initialize GenAI client
try:
    genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID)
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

    temp_image_path = "temp_image.png"

    try:
        # Convert base64 → PNG bytes → GenAI image
        img_bytes = base64.b64decode(base64_image_data.split(",", 1)[1])
        pil_image = PIL.Image.open(io.BytesIO(img_bytes))

        # Save image temporarily to pass it as a file for Veo 3.1 video generation
        pil_image.save(temp_image_path)

        # -----------------------------
        #   VIDEO GEN WITH VEO 3.1
        # -----------------------------
        enhance_prompt = True  # Set based on your preference
        generate_audio = True  # Set based on your preference

        # Generate the video with Veo 3.1 using Vertex AI Client
        operation = genai_client.models.generate_videos(
            model=MODEL_ID_VIDEO,
            prompt=video_prompt,
            image=types.Image.from_file(location=temp_image_path),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",  # Adjust to your desired aspect ratio
                number_of_videos=1,
                duration_seconds=6,   # Adjust to your desired video duration
                resolution="1080p",   # Adjust to your desired resolution
                person_generation="allow_adult",  # Adjust based on your use case
                enhance_prompt=enhance_prompt,
                generate_audio=generate_audio,
            ),
        )

        # Poll the operation to check when it's done
        print("Polling Veo video generation...")
        while not operation.done:
            time.sleep(15)  # Wait for the operation to finish
            operation = genai_client.operations.get(operation)
            print(operation)  # For debugging

        if not operation.response or not operation.response.generated_videos:
            raise ValueError("Veo returned no videos.")

        # Check for video result and display it (video_bytes or video URI)
        video = operation.response.generated_videos[0]
        video_url = ""

        # --- FIX STARTS HERE ---
        # The 'video' object is a wrapper. We must check for 'video_bytes' property inside 'video.video'
        
        # 1. Check if video bytes are present
        if hasattr(video.video, "video_bytes") and video.video.video_bytes:
            video_file_name = f"video_{uuid.uuid4()}.mp4"
            video_path = os.path.join(LOCAL_VIDEO_DIR, video_file_name)
            
            with open(video_path, "wb") as out_file:
                out_file.write(video.video.video_bytes)  # Access the actual bytes

            print(f"Video saved locally at {video_path}")
            video_url = f"/static/generated_videos/{video_file_name}"

        # 2. Check if video URI is present (fallback for some cloud storage configs)
        elif hasattr(video.video, "uri") and video.video.uri:
            video_url = video.video.uri
            print(f"Video available at cloud URL: {video_url}")
        
        else:
            raise ValueError("Video result contained neither bytes nor URI.")
        # --- FIX ENDS HERE ---

        # Return the video URL to the frontend
        return jsonify({"generated_video_url": video_url})

    except Exception as e:
        print(f"Video generation error: {e}")
        return jsonify({"error": f"Failed to generate video: {e}"}), 500
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass


# --------------------------------------------------------
#                    HEALTH CHECK
# --------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "genai_client": genai_client is not None,
        "image_model": MODEL_ID_IMAGE,
        "video_model": MODEL_ID_VIDEO,
    })


# --------------------------------------------------------
#               GENERATED FILE LISTING
# --------------------------------------------------------
@app.route("/history/images")
def list_images():
    files = utils.list_generated_files(LOCAL_IMAGE_DIR, extensions=(".png", ".jpg"))
    return jsonify({"images": files, "count": len(files)})


@app.route("/history/videos")
def list_videos():
    files = utils.list_generated_files(LOCAL_VIDEO_DIR, extensions=(".mp4",))
    return jsonify({"videos": files, "count": len(files)})


# --------------------------------------------------------
#               DOWNLOAD GENERATED FILES
# --------------------------------------------------------
@app.route("/download/image/<filename>")
def download_image(filename):
    safe_dir = os.path.abspath(LOCAL_IMAGE_DIR)
    full_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not full_path.startswith(safe_dir):
        abort(403)
    return send_from_directory(safe_dir, filename, as_attachment=True)


@app.route("/download/video/<filename>")
def download_video(filename):
    safe_dir = os.path.abspath(LOCAL_VIDEO_DIR)
    full_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not full_path.startswith(safe_dir):
        abort(403)
    return send_from_directory(safe_dir, filename, as_attachment=True)


# --------------------------------------------------------
#               CLEANUP OLD GENERATED FILES
# --------------------------------------------------------
@app.route("/admin/cleanup", methods=["POST"])
def cleanup():
    """Remove generated files older than 24 hours from both asset directories."""
    removed_images = utils.cleanup_old_files(LOCAL_IMAGE_DIR)
    removed_videos = utils.cleanup_old_files(LOCAL_VIDEO_DIR)
    return jsonify({
        "removed_images": removed_images,
        "removed_videos": removed_videos,
    })


# --------------------------------------------------------
#                        RUN FLASK
# --------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)