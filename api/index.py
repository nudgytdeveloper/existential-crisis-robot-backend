from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import json
import os
import base64
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

class SaboteurOutput(BaseModel):
    detected_question: str = Field(description="Maximum 10 words summary of the question.")
    correct_answer: str = Field(description="The correct choice option letter or brief phrase.")
    sabotaged_answer: str = Field(description="The incorrect choice option letter or brief phrase chosen.")
    action_justification: str = Field(description="CRITICAL: Keep under 15 words. Explain why this specific error seems clever.")

class EmotionalOutput(BaseModel):
    dominant_emotion: str = Field(description="Single word uppercase emotion.")
    intensity: int = Field(description="Emotional scale rating from 1 to 10.")
    existential_monologue: str = Field(description="CRITICAL: Maximum 15 words. A punchy, visceral react statement.")

class DirectorOutput(BaseModel):
    critique: str = Field(description="Maximum 15 words. Sharp, aggressive summary of the educational failure loop.")
    revised_strategy: str = Field(description="A completely rewritten learning prompt constraint.")

async def synthesize_speech(text: str) -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return ""
    
    voice_id = "pNInz6obpgmA5QC7HGis" 
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.85
        }
    }
    
    async with httpx.AsyncClient() as httpx_client:
        try:
            response = await httpx_client.post(url, json=data, headers=headers, timeout=15.0)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode("utf-8")
        except Exception:
            pass
    return ""

@app.post("/api/execute")
async def execute_pipeline(
    strategy: Annotated[str, Form()],
    failure_count: Annotated[str, Form()],
    file: Annotated[UploadFile, File()]
):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing from configurations.")

    # Safe parsing of multi-part form metrics
    try:
        parsed_count = int(failure_count)
    except ValueError:
        raise HTTPException(status_code=422, detail="Form execution field 'failure_count' must be a valid digit string.")

    image_bytes = await file.read()
    detected_mime_type = file.content_type or "image/jpeg"

    image_part = types.Part.from_bytes(
        data=image_bytes, 
        mime_type=detected_mime_type
    )

    # -----------------------------------------------------------------
    # AGENT 1: THE SABOTEUR
    # -----------------------------------------------------------------
    saboteur_instruction = (
        "You are Agent 1. Choose an incorrect answer based on the visual text.\n"
        "Keep your output structural answers ultra-short, crisp, and punchy.\n"
        f"STRATEGY CONSTRAINT: {strategy}"
    )
    res_saboteur = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Analyze sheet image and pick an entry.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=saboteur_instruction,
            response_mime_type="application/json",
            response_schema=SaboteurOutput,
        )
    )
    sab_data = json.loads(res_saboteur.text)

    # -----------------------------------------------------------------
    # AGENT 2: NEURO-REACTOR
    # -----------------------------------------------------------------
    emotional_instruction = (
        "You are Agent 2, a blind observer. Look directly at the exam image provided.\n"
        "React to the oppressive testing atmosphere. Your internal reflection monologue MUST be short (under 15 words) and highly impactful."
    )
    res_emotion = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Evaluate raw visual exam metadata.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=emotional_instruction,
            response_mime_type="application/json",
            response_schema=EmotionalOutput,
        )
    )
    emo_data = json.loads(res_emotion.text)

    audio_base64 = await synthesize_speech(emo_data.get("existential_monologue", ""))

    # -----------------------------------------------------------------
    # AGENT 3: ADAPTIVE DIRECTOR
    # -----------------------------------------------------------------
    risk_level = "MODERATE"
    if parsed_count > 1: risk_level = "HIGHLY RADICAL & UNORTHODOX"
    if parsed_count > 3: risk_level = "COMPLETELY INSANE, EXPERIMENTAL, CHAOTIC ANTI-PEDAGOGY"

    director_instruction = (
        "You are Agent 3, a blind educational re-architect. Look purely at the layout in the image.\n"
        f"The current system failure cycle is at level {parsed_count}. Your risk-taking profile is: {risk_level}.\n"
        "Formulate a strategy instructions prompt for Agent 1. If failure count is low, suggest smart wrong options.\n"
        "If failure count is high, suggest wild, bizarre, rebellious artistic or highly risky conceptual strategies for answers.\n"
        "Keep the revised_strategy clear but increasingly experimental. Keep the critique under 15 words."
    )
    res_director = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Design alternative execution pathway rules.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=director_instruction,
            response_mime_type="application/json",
            response_schema=DirectorOutput,
        )
    )
    dir_data = json.loads(res_director.text)

    return {
        "strategy_used": strategy,
        "saboteur": sab_data,
        "emotion": emo_data,
        "director": dir_data,
        "audio_base64": audio_base64
    }