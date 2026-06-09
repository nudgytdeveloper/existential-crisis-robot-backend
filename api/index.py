from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = FastAPI()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializing Client - reads GEMINI_API_KEY automatically from environment
client = genai.Client()

# Pydantic Structural Schemas
class SaboteurOutput(BaseModel):
    detected_question: str
    correct_answer: str
    sabotaged_answer: str
    action_justification: str

class EmotionalOutput(BaseModel):
    dominant_emotion: str = Field(description="Single word uppercase emotion.")
    intensity: int = Field(description="Emotional scale rating from 1 to 10.")
    existential_monologue: str = Field(description="Internal reaction to an exam failure scenario.")

class DirectorOutput(BaseModel):
    critique: str = Field(description="Analysis of standard testing gaps.")
    revised_strategy: str = Field(description="An updated strategy string for future evaluation cycles.")

@app.post("/api/execute")
async def execute_pipeline(
    strategy: str = Form(...),
    file: UploadFile = File(...)
):
    # Verify API Key exists in Vercel Runtime Env
    if not os.environ.get("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY is not configured in environment variables."}

    image_bytes = await file.read()
    detected_mime_type = file.content_type or "image/jpeg"

    image_part = types.Part.from_bytes(
        data=image_bytes, 
        mime_type=detected_mime_type
    )

    # -----------------------------------------------------------------
    # AGENT 1: THE SABOTEUR (Completely Blind)
    # -----------------------------------------------------------------
    saboteur_instruction = (
        "You are Agent 1, working entirely alone. Your unique goal is to view the image and choose an incorrect answer.\n"
        "You have no knowledge of any other agents in the system.\n"
        f"STRATEGY CONSTRAINT: {strategy}"
    )
    res_saboteur = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Examine image and fail according to instructions.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=saboteur_instruction,
            response_mime_type="application/json",
            response_schema=SaboteurOutput,
        )
    )
    sab_data = json.loads(res_saboteur.text)

    # -----------------------------------------------------------------
    # AGENT 2: EMOTIONAL REACTOR (Completely Blind)
    # -----------------------------------------------------------------
    emotional_instruction = (
        "You are Agent 2, an isolated observer. You do not know what choices Agent 1 or Agent 3 made.\n"
        "Look at the uploaded exam sheet image. Based purely on the content, context, and the concept of high-stakes academic pressure (like the Singapore PSLE),\n"
        "output an intense independent cognitive/emotional state regarding potential failure, and an intensity level from 1-10."
    )
    res_emotion = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Evaluate the emotional weight of this exam context.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=emotional_instruction,
            response_mime_type="application/json",
            response_schema=EmotionalOutput,
        )
    )
    emo_data = json.loads(res_emotion.text)

    # -----------------------------------------------------------------
    # AGENT 3: ADAPTIVE DIRECTOR (Completely Blind)
    # -----------------------------------------------------------------
    director_instruction = (
        "You are Agent 3, an independent system architect. You have no visibility into Agent 1's actions or Agent 2's feelings.\n"
        "Look purely at the exam material provided in the image. Design a completely revised learning directive or abstract pedagogical constraint "
        "to handle this educational topic differently in the next iteration."
    )
    res_director = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Generate an independent educational strategy for this content.", image_part],
        config=types.GenerateContentConfig(
            system_instruction=director_instruction,
            response_mime_type="application/json",
            response_schema=DirectorOutput,
        )
    )
    dir_data = json.loads(res_director.text)

    # Combined payload package returned to UI layer
    return {
        "strategy_used": saboteur_instruction,
        "saboteur": sab_data,
        "emotion": emo_data,
        "director": dir_data
    }