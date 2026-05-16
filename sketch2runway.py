import os
import io
import time
import argparse
import datetime
from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt
from google import genai
from google.genai import types
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv(".env")

# Model IDs — override via --image_model / --video_model if needed
DEFAULT_IMAGE_MODEL = "gemini-2.0-flash-exp-image-generation"
DEFAULT_VIDEO_MODEL = "veo-3.0-generate-preview"

SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

SYSTEM_INSTRUCTIONS = (
    "You are an AI assistant specializing in transforming hand-drawn sketches "
    "into photorealistic fashion images. Your task is to take the user's uploaded "
    "sketch and convert it into real photographs as if taken from a DSLR HD camera."
)


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def build_image_client(api_key):
    return genai.Client(api_key=api_key)


def build_video_client(project_id, location="us-central1"):
    return genai.Client(vertexai=True, project=project_id, location=location)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path):
    img = Image.open(path)
    print(f"Loaded image: {path}  ({img.width}x{img.height}, mode={img.mode})")
    return img


def display_image(pil_image, title="Image"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 9))
    ax.imshow(pil_image)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_image_from_response(response, output_path):
    """Extract the first image part from a Gemini response and save it."""
    for part in response.candidates[0].content.parts:
        if part.text:
            print("Model says:", part.text.strip())
        elif part.inline_data:
            img = Image.open(io.BytesIO(part.inline_data.data))
            img.save(output_path)
            print(f"Image saved: {output_path}")
            return img
    raise ValueError("No image found in Gemini response.")


# ---------------------------------------------------------------------------
# Step 5: Sketch → photorealistic image
# ---------------------------------------------------------------------------

def sketch_to_image(client, sketch_path, prompt, model_id=DEFAULT_IMAGE_MODEL, output_path="generated_image.png"):
    img = load_image(sketch_path)
    print(f"Generating photorealistic image from sketch: {sketch_path}")

    response = client.models.generate_content(
        model=model_id,
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            temperature=0.5,
            safety_settings=SAFETY_SETTINGS,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    return save_image_from_response(response, output_path)


# ---------------------------------------------------------------------------
# Step 6: Edit the generated photo
# ---------------------------------------------------------------------------

def edit_image(client, image_path, edit_prompt, model_id=DEFAULT_IMAGE_MODEL, output_path="edited_image.png"):
    img = load_image(image_path)
    print(f"Editing image with prompt: {edit_prompt!r}")

    response = client.models.generate_content(
        model=model_id,
        contents=[edit_prompt, img],
        config=types.GenerateContentConfig(
            temperature=0.5,
            safety_settings=SAFETY_SETTINGS,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    return save_image_from_response(response, output_path)


# ---------------------------------------------------------------------------
# Step 7: Upload image to GCS
# ---------------------------------------------------------------------------

def upload_to_gcs(local_path, bucket_name, destination_blob=None):
    if destination_blob is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        destination_blob = f"images/{ts}_{Path(local_path).name}"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{bucket_name}/{destination_blob}"
    print(f"Uploaded to GCS: {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Step 8: Generate video with Veo 3
# ---------------------------------------------------------------------------

def generate_video(client, image_gcs_uri, prompt, bucket_name, model_id=DEFAULT_VIDEO_MODEL,
                   duration_seconds=8, generate_audio=True, enhance_prompt=True):
    output_gcs = f"gs://{bucket_name}/videos"

    print(f"Submitting video generation job — model: {model_id}")
    operation = client.models.generate_videos(
        model=model_id,
        prompt=prompt,
        image=types.Image(gcs_uri=image_gcs_uri, mime_type="image/png"),
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            output_gcs_uri=output_gcs,
            number_of_videos=1,
            duration_seconds=duration_seconds,
            person_generation="allow_adult",
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
        ),
    )

    print("Polling video generation (this takes a few minutes)...")
    while not operation.done:
        time.sleep(6)
        operation = client.operations.get(operation)
        print("  ... still generating ...")

    if not operation.response:
        raise RuntimeError("Video generation operation completed with no response.")

    video_uri = operation.result.generated_videos[0].video.uri
    print(f"Video ready: {video_uri}")
    return video_uri


# ---------------------------------------------------------------------------
# Download video from GCS to local
# ---------------------------------------------------------------------------

def download_from_gcs(gcs_uri, local_path):
    storage_client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded video to: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sketch2Runway: Convert a fashion sketch to a photorealistic image and runway video."
    )
    parser.add_argument("--sketch", required=True, help="Path to the input sketch image (PNG/JPG).")
    parser.add_argument(
        "--prompt",
        default="Convert this input sketch of an outfit into a catalogue photograph that shows a beautiful woman model wearing it.",
        help="Prompt for sketch-to-image generation.",
    )
    parser.add_argument("--edit_prompt", default=None, help="Optional prompt to edit the generated image.")
    parser.add_argument(
        "--video_prompt",
        default="The fashion model walks toward the camera with a smile.",
        help="Prompt for video generation with Veo 3.",
    )
    parser.add_argument("--image_model", default=DEFAULT_IMAGE_MODEL, help="Gemini image generation model ID.")
    parser.add_argument("--video_model", default=DEFAULT_VIDEO_MODEL, help="Veo video generation model ID.")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save generated files.")
    parser.add_argument("--skip_video", action="store_true", help="Stop after image generation, skip video.")
    parser.add_argument("--duration", type=int, default=8, help="Video duration in seconds (default: 8).")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio generation in the video.")
    parser.add_argument("--display", action="store_true", help="Display images with matplotlib during the run.")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    gcp_project = os.environ.get("GCP_PROJECT_ID")
    gcs_bucket = os.environ.get("GCS_BUCKET")

    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your .env file.")

    os.makedirs(args.output_dir, exist_ok=True)

    image_client = build_image_client(api_key)

    # Step 5: sketch → photo
    generated_path = os.path.join(args.output_dir, "generated_image.png")
    generated = sketch_to_image(image_client, args.sketch, args.prompt, args.image_model, generated_path)
    if args.display:
        display_image(generated, "Generated Image")

    # Step 6: optional edit
    final_image_path = generated_path
    if args.edit_prompt:
        edited_path = os.path.join(args.output_dir, "edited_image.png")
        edited = edit_image(image_client, generated_path, args.edit_prompt, args.image_model, edited_path)
        final_image_path = edited_path
        if args.display:
            display_image(edited, "Edited Image")

    if args.skip_video:
        print("Skipping video generation (--skip_video).")
        return

    if not gcp_project or not gcs_bucket:
        print("GCP_PROJECT_ID or GCS_BUCKET not set — skipping video generation.")
        return

    # Step 7: upload to GCS
    gcs_uri = upload_to_gcs(final_image_path, gcs_bucket)

    # Step 8: generate video
    video_client = build_video_client(gcp_project)
    video_uri = generate_video(
        video_client,
        gcs_uri,
        args.video_prompt,
        gcs_bucket,
        model_id=args.video_model,
        duration_seconds=args.duration,
        generate_audio=not args.no_audio,
    )

    # Download video locally
    local_video = os.path.join(args.output_dir, "runway_video.mp4")
    download_from_gcs(video_uri, local_video)
    print(f"\nAll done. Final video: {local_video}")


if __name__ == "__main__":
    main()
