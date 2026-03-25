import io
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Voice - ElevenLabs TTS API", version="2.0.0")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

# CORS - allow all origins (configure for production)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == "*" else allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Curated voices - clear and easy to understand
VOICES = [
    {
        "id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Sarah",
        "gender": "Female",
        "description": "Mature, Reassuring, Confident",
    },
    {
        "id": "cjVigY5qzO86Huf0OWal",
        "name": "Eric",
        "gender": "Male",
        "description": "Smooth, Trustworthy",
    },
    {
        "id": "Xb7hH8MSUJpSbSDYk0k2",
        "name": "Alice",
        "gender": "Female",
        "description": "Clear, Engaging Educator",
    },
    {
        "id": "onwK4e9ZLuTAKqWW03F9",
        "name": "Daniel",
        "gender": "Male",
        "description": "Steady Broadcaster",
    },
]


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # Default: Sarah
    speed: float = 1.0  # 0.5 to 2.0


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/voices")
async def list_voices():
    """Return curated voice list."""
    return {"voices": VOICES}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Convert text to speech using ElevenLabs and return MP3 audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")

    try:
        url = f"{ELEVENLABS_BASE}/text-to-speech/{req.voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        body = {
            "text": req.text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": req.speed,
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            detail = response.text
            raise HTTPException(status_code=response.status_code, detail=detail)

        audio_buffer = io.BytesIO(response.content)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5050"))
    uvicorn.run(app, host=host, port=port)
