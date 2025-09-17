import os
import json
from typing import Literal
from .models import LLMResponse

import google.generativeai as genai

GENERATOR_MODEL = "gemini-1.5-flash-latest"

def _configure_genai():
    key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not key:
        raise EnvironmentError("Set GOOGLE_GENAI_API_KEY for online mode.")
    genai.configure(api_key=key)

class LLMWrapper:
    def __init__(self, mode: Literal["online", "offline"]):
        self.mode = mode
        if mode == "online":
            _configure_genai()

    def _call_offline_model(self, prompt: str) -> str:
        mock = {
            "answer": "⚠️ Offline mode – no external call.",
            "thoughts": "Echoing prompt for debugging.",
            "citations": [],
        }
        return json.dumps(mock)

    def _call_online_model(self, prompt: str) -> str:
        model = genai.GenerativeModel(GENERATOR_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 2048},
        )
        return resp.text

    def generate(self, prompt: str) -> LLMResponse:
        raw = self._call_online_model(prompt) if self.mode == "online" else self._call_offline_model(prompt)
        try:
            return LLMResponse(**json.loads(raw))
        except Exception as e:
            return LLMResponse(answer=f"Malformed JSON from LLM: {e}\n\nRaw:\n{raw}", citations=[])

def get_system_prompt(role: Literal["manager", "field"], context: str = "") -> str:
    base = (
        "You are a helpful assistant for a humanitarian aid organization. "
        "Always respond with JSON containing keys: 'answer', 'thoughts', 'citations'."
    )
    if role == "manager":
        role_p = "You are speaking to a Manager. Provide detailed, strategic advice."
    else:
        role_p = "You are speaking to a Field Member. Provide concise, actionable advice for low‑connectivity devices."
    return f"{base}\n\n{role_p}\n\n{context}"
