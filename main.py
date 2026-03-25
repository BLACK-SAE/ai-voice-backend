import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import edge_tts

load_dotenv()

app = FastAPI(title="AI Voice - Edge TTS API", version="2.0.0")

# CORS - allow all origins (configure for production)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == "*" else allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Curated Nigerian English voices
NIGERIAN_VOICES = [
    {
        "id": "en-NG-AbeoNeural",
        "name": "Abeo",
        "gender": "Male",
        "description": "Nigerian Male",
    },
    {
        "id": "en-NG-EzinneNeural",
        "name": "Ezinne",
        "gender": "Female",
        "description": "Nigerian Female",
    },
]

# Additional clear English voices Nigerians can easily understand
EXTRA_VOICES = [
    {
        "id": "en-US-GuyNeural",
        "name": "Guy",
        "gender": "Male",
        "description": "American Male — Clear & Neutral",
    },
    {
        "id": "en-GB-SoniaNeural",
        "name": "Sonia",
        "gender": "Female",
        "description": "British Female — Clear & Steady",
    },
]


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "en-NG-AbeoNeural"
    speed: float = 1.0  # 0.5 to 2.0


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/voices")
async def list_voices():
    """Return curated voice list."""
    return {"voices": NIGERIAN_VOICES + EXTRA_VOICES}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Convert text to speech using Edge-TTS and return MP3 audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Convert speed float to edge-tts rate string
        pct = round((req.speed - 1) * 100)
        rate = f"+{pct}%" if pct >= 0 else f"{pct}%"

        communicate = edge_tts.Communicate(
            text=req.text,
            voice=req.voice_id,
            rate=rate,
        )

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
