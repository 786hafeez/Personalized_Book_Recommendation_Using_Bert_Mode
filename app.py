from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from diffusers import DiffusionPipeline
import os
import uuid
import torch

# Initialize FastAPI
app = FastAPI(title="ImaginAItion API", version="1.1.0")

# Store prompt history in memory
history = []

# Load Stable Diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# Image directory
IMAGE_DIR = "generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Serve images as static files
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

@app.get("/")
async def root():
    return {"message": "Welcome to the ImaginAItion API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

@app.get("/generate", response_class=FileResponse)
async def generate_image(prompt: str = Query(..., description="Text prompt for image generation")):
    try:
        image = pipe(prompt).images[0]
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(IMAGE_DIR, filename)
        image.save(path)

        # Save prompt and image URL path in memory
        history.append({"prompt": prompt, "image_path": f"/images/{filename}"})

        return FileResponse(path, media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/history")
async def get_history():
    return history

@app.delete("/history")
async def clear_history():
    history.clear()
    # Also delete all files in IMAGE_DIR
    for f in os.listdir(IMAGE_DIR):
        os.remove(os.path.join(IMAGE_DIR, f))
    return {"message": "Prompt history cleared!"}
