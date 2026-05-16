import os
import io
import base64
import uuid
import time
from datetime import datetime
from pathlib import Path

import PIL.Image

# Max dimensions Gemini image generation accepts for the input sketch
MAX_INPUT_WIDTH = 2048
MAX_INPUT_HEIGHT = 2048


def decode_base64_image(base64_string):
    """Strip data URI prefix if present and decode to bytes."""
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    return base64.b64decode(base64_string)


def bytes_to_pil(image_bytes):
    """Convert raw image bytes to a PIL Image."""
    return PIL.Image.open(io.BytesIO(image_bytes))


def pil_to_base64(pil_image, fmt="PNG"):
    """Encode a PIL Image to a base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_if_needed(pil_image, max_width=MAX_INPUT_WIDTH, max_height=MAX_INPUT_HEIGHT):
    """Downscale the image only if it exceeds the max allowed dimensions.

    Gemini rejects images larger than 2048px on either side for image generation.
    Aspect ratio is always preserved.
    """
    w, h = pil_image.size
    if w <= max_width and h <= max_height:
        return pil_image

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return pil_image.resize((new_w, new_h), PIL.Image.LANCZOS)


def validate_image_bytes(image_bytes, min_size_bytes=100):
    """Raise ValueError if the image data looks malformed or too small."""
    if not image_bytes or len(image_bytes) < min_size_bytes:
        raise ValueError(f"Image data is too small ({len(image_bytes)} bytes).")
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception as exc:
        raise ValueError(f"Invalid image data: {exc}")


def save_image_locally(image_bytes, output_dir, filename=None):
    """Save image bytes to disk and return the relative URL path."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"image_{uuid.uuid4()}.png"
    full_path = os.path.join(output_dir, filename)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    return full_path, filename


def get_image_metadata(pil_image):
    """Return a dict of basic image metadata."""
    return {
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
        "format": pil_image.format,
    }


def cleanup_old_files(directory, max_age_seconds=86400, extensions=(".mp4", ".png", ".jpg")):
    """Delete files older than max_age_seconds from a directory.

    Keeps the folder tidy when the server runs for a long time without restarts.
    Default retention is 24 hours.
    """
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    for fname in os.listdir(directory):
        if not fname.lower().endswith(extensions):
            continue
        fpath = os.path.join(directory, fname)
        try:
            age = now - os.path.getmtime(fpath)
            if age > max_age_seconds:
                os.remove(fpath)
                removed += 1
        except OSError:
            pass
    return removed


def human_readable_size(num_bytes):
    """Convert a byte count to a human-readable string (KB / MB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / 1024 ** 2:.1f} MB"


def list_generated_files(directory, extensions=(".mp4", ".png", ".jpg")):
    """Return a list of dicts describing files in a generated-assets directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for fname in sorted(os.listdir(directory), reverse=True):
        if not fname.lower().endswith(extensions):
            continue
        fpath = os.path.join(directory, fname)
        stat = os.stat(fpath)
        files.append({
            "filename": fname,
            "size": human_readable_size(stat.st_size),
            "created": datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M UTC"),
        })
    return files
