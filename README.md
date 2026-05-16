# Fashion Sketch2Runway

Sketch2Runway turns a hand-drawn outfit sketch into a photorealistic fashion photo and then animates it into a runway-style video — all powered by Google's Gemini image generation and Veo 3.1 video generation models.

## Results:
<img src="https://github.com/NSTiwari/Sketch2Runway/blob/main/static/images/sketch2runway.gif"/>

The project has two interfaces:
- **Flask web app** — draw your sketch in the browser, generate a photorealistic image, edit it with a text prompt, then animate it into a 6-second runway video using Veo 3.1, all from one page.
- **Python CLI script** (`sketch2runway.py`) — same pipeline from the command line, suitable for batch use or automation with a GCS bucket + Veo 3 Vertex AI endpoint.


## How it works

**Step 1 — Sketch canvas**

The web app's canvas is built with the HTML5 Canvas API. It supports freehand drawing, color picker, line thickness control, draw/erase toggle, and touch input. When the user clicks Generate, the canvas content is serialized to a base64 PNG and POSTed to the Flask backend.

**Step 2 — Sketch → photorealistic image (Gemini)**

The backend (`app.py`) sends the base64 sketch image and a text prompt to Gemini via the `google-genai` SDK. The model is called with `response_modalities=["TEXT", "IMAGE"]` and a 2:3 aspect ratio to match a portrait fashion photograph. Gemini's understanding of clothing structure lets it fill in realistic textures, lighting, and background from just a rough outline.

**Step 3 — Image editing (Gemini)**

The generated image is passed back into the same Gemini endpoint with a new edit prompt (e.g., "Add a white bead necklace, change background to a beach"). The model edits in-place while preserving the model, outfit structure, and lighting.

**Step 4 — Image → video (Veo 3.1)**

The edited image is sent to Veo 3.1 (`veo-3.1-generate-preview`) via the Vertex AI client. The API call specifies the input image as the first frame, plus a motion prompt describing how the model should move. Veo generates a 1080p, 16:9, 6-second video with optional ambient audio. Since video generation is asynchronous, the backend polls the operation every 15 seconds until it completes, then saves the `.mp4` locally to `/static/generated_videos/` and returns a URL to the frontend.

**Two backends**

- `app.py` — uses `genai.Client(vertexai=True, project=GCP_PROJECT_ID)` for both image and video, so both go through Vertex AI. Recommended for production.
- `app_gemini_api.py` — uses a plain Gemini API key (`genai.Client(api_key=...)`) for image generation, without Vertex AI. Simpler to set up for local testing.

**Utilities (`utils.py`)**

Shared helpers used by `app.py`: base64 decode/encode, PIL image validation, input dimension capping (Gemini rejects inputs >2048px), local file save, a JSON-based generation history log, and a cleanup function that deletes generated files older than 24 hours.


## Project structure

```
Sketch2Runway/
├── app.py                      # Flask app — Vertex AI backend (image + video)
├── app_gemini_api.py           # Flask app — Gemini API key backend (image + video)
├── sketch2runway.py            # CLI script — full pipeline with GCS + Veo 3 Vertex
├── utils.py                    # Shared image/file utilities
├── requirements.txt
├── .env                        # API keys and model config (not committed)
├── templates/
│   └── index.html              # Single-page UI: canvas, carousel steps, JS fetch
└── static/
    ├── images/
    │   └── sketch2runway.gif
    ├── generated_images/       # Saved generated/edited images (auto-created)
    └── generated_videos/       # Saved Veo output MP4s (auto-created)
```


## Prerequisites

1. **Google Cloud SDK** installed for Vertex AI authentication:
   ```bash
   gcloud init
   gcloud auth application-default login
   ```

2. **Veo 3 access** — available for public preview. Ensure the Vertex AI API is enabled on your GCP project.

3. **GCS bucket** — required only for the CLI script and the Vertex AI backend. The Gemini API backend writes videos locally.


## Run the Web App

```bash
git clone https://github.com/NSTiwari/Sketch2Runway.git
cd Sketch2Runway
pip install -r requirements.txt
```

Edit `.env` and set:
```
GOOGLE_API_KEY=your_gemini_or_vertex_api_key
GCP_PROJECT_ID=your_gcp_project_id
IMAGE_GEN_MODEL=gemini-2.5-flash-preview-05-20
VIDEO_GEN_MODEL=veo-3.1-generate-preview
```

Then start the server:
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

For the simpler Gemini API version (no Vertex AI):
```bash
python app_gemini_api.py
```


## Run the CLI Script

Sketch-to-image only (no video):
```bash
python sketch2runway.py \
  --sketch my_sketch.png \
  --prompt "Convert this sketch into a fashion catalogue photo of a woman model." \
  --skip_video \
  --display
```

Full pipeline with image editing and Veo 3 video:
```bash
python sketch2runway.py \
  --sketch my_sketch.png \
  --prompt "Convert this sketch into a fashion catalogue photo of a woman model." \
  --edit_prompt "Add a white bead necklace and change the background to a beach." \
  --video_prompt "The fashion model walks toward the camera with a confident smile." \
  --duration 8 \
  --output_dir outputs/
```


## API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve the web UI |
| POST | `/generate-image` | Sketch/image → Gemini image generation |
| POST | `/generate-video` | Image → Veo 3.1 video generation |
| GET | `/health` | Service health check |
| GET | `/history/images` | List generated images with metadata |
| GET | `/history/videos` | List generated videos with metadata |
| GET | `/download/image/<filename>` | Download a generated image |
| GET | `/download/video/<filename>` | Download a generated video |
| POST | `/admin/cleanup` | Delete generated files older than 24h |


## Tutorials
- [YouTube tutorial](https://youtu.be/9Bo5sEVoVQk)
- [Blog post](https://margaretmz.medium.com/fashion-sketch2runway-with-gemini2flash-veo3-ced1e2776fea)
