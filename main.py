import io
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import edge_tts

load_dotenv()

app = FastAPI(title="AI Voice - Edge TTS API", version="1.0.0")

# CORS - allow all origins (configure for production)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == "*" else allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "en-US-GuyNeural"
    speed: str = "+0%"    # e.g. "+20%", "-10%"
    pitch: str = "+0Hz"   # e.g. "+5Hz", "-10Hz"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


ACCENT_LABELS = {
    "en-US": "American",
    "en-GB": "British",
    "en-AU": "Australian",
    "en-IN": "Indian",
    "en-ZA": "South African",
    "en-KE": "Kenyan",
    "en-NG": "Nigerian",
    "en-TZ": "Tanzanian",
}


@app.get("/api/voices")
async def list_voices():
    """List English-only Edge-TTS voices, grouped by accent."""
    try:
        voices_list = await edge_tts.list_voices()
        grouped = {}
        for v in voices_list:
            locale = v["Locale"]
            # Filter to English voices only
            if not locale.startswith("en-"):
                continue
            label = ACCENT_LABELS.get(locale, locale)
            if label not in grouped:
                grouped[label] = []
            grouped[label].append({
                "name": v["ShortName"],
                "locale": locale,
                "gender": v["Gender"],
                "friendlyName": v["FriendlyName"],
            })
        return {"voices": grouped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Convert text to speech and return MP3 audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        communicate = edge_tts.Communicate(
            text=req.text,
            voice=req.voice,
            rate=req.speed,
            pitch=req.pitch,
        )

        # Collect audio data into a buffer
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5050"))
    uvicorn.run(app, host=host, port=port)
