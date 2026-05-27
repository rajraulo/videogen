"""
AudioProcessor — Whisper speech transcription + librosa mood analysis.
Supports Odia (or), Telugu (te), and all other Whisper languages.
"""

import numpy as np
from typing import Optional, Tuple

_whisper_model = None
_whisper_model_size = None

MOOD_STYLE_MAP: dict[str, str] = {
    "energetic": "fast motion, dynamic movement, vibrant saturated colors, high energy",
    "calm":      "slow motion, serene landscape, soft diffused lighting, tranquil",
    "dark":      "dramatic shadows, moody low-key lighting, noir atmosphere, desaturated",
    "happy":     "bright sunny colors, cheerful playful movement, golden hour warmth",
    "epic":      "cinematic grandeur, sweeping wide-angle shots, dramatic scale, heroic",
}

# Whisper language codes for supported languages
LANGUAGE_CODES: dict[str, str] = {
    "auto":   None,      # Whisper auto-detects
    "odia":   "or",
    "telugu": "te",
    "hindi":  "hi",
    "english": "en",
}


def _get_whisper_model(size: str = "small"):
    """Lazy-load Whisper model. 'small' recommended for Odia/Telugu accuracy."""
    global _whisper_model, _whisper_model_size
    if _whisper_model is None or _whisper_model_size != size:
        import whisper
        print(f"[AudioProcessor] Loading Whisper '{size}' model…")
        _whisper_model = whisper.load_model(size)
        _whisper_model_size = size
        print("[AudioProcessor] Whisper ready.")
    return _whisper_model


def transcribe_speech(
    audio_path: str,
    language: Optional[str] = None,
    model_size: str = "small",
) -> str:
    """
    Transcribe speech to text using Whisper.

    Args:
        audio_path:  Path to the audio file (m4a, mp3, wav, etc.)
        language:    ISO 639-1 code — None = auto-detect. Use "or" for Odia, "te" for Telugu.
        model_size:  Whisper model size ('small' is the minimum recommended for Indic languages).
    """
    model = _get_whisper_model(model_size)
    result = model.transcribe(audio_path, language=language, fp16=False)
    text = result["text"].strip()
    detected = result.get("language", "unknown")
    print(f"[AudioProcessor] Transcribed ({detected}): {text[:80]}")
    return text


def analyze_audio_mood(audio_path: str) -> Tuple[str, str]:
    """
    Analyze audio energy/mood with librosa.
    Returns (mood_key, style_description).
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=None, duration=30.0, mono=True)
    if len(y) == 0:
        return "epic", MOOD_STYLE_MAP["epic"]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    if tempo > 140 and rms > 0.05:
        mood = "energetic"
    elif tempo > 115 and centroid > 3000:
        mood = "happy"
    elif tempo < 70 or (rms < 0.015 and centroid < 1500):
        mood = "dark"
    elif tempo < 90 and rms < 0.04:
        mood = "calm"
    else:
        mood = "epic"

    print(f"[AudioProcessor] Mood: {mood} (tempo={tempo:.0f}, rms={rms:.3f})")
    return mood, MOOD_STYLE_MAP[mood]


def build_enriched_prompt(transcript: str, music_style: Optional[str] = None) -> str:
    """Merge transcript + optional music mood into a generation-ready prompt."""
    if music_style:
        return f"{transcript}, {music_style}"
    return transcript
