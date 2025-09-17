from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class LoginRequest(BaseModel):
    role: Literal["manager", "field"]

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class Citation(BaseModel):
    document: str
    page: int
    text: str

class LLMResponse(BaseModel):
    answer: str
    thoughts: Optional[str] = None
    citations: List[Citation] = []

class PlanRequest(BaseModel):
    crisis_type: str = Field(..., example="Flash Flood")
    location: str = Field(..., example="Coastal region, 10km from Port City")
    population: int = Field(..., example=5000)
    constraints: str = Field(..., example="Limited road access, power outage expected for 72 hours.")
    selected_standards: List[str] = Field(..., example=["Sphere", "CHS"])

class PlanResponse(LLMResponse):
    pass

class ChatRequest(BaseModel):
    role: Literal["manager", "field"]
    message: str
    mode: Literal["online", "offline"]

class ChatResponse(LLMResponse):
    pass

class Task(BaseModel):
    id: int
    title: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    assigned_to: str

class TaskCreate(BaseModel):
    title: str
    description: str
    assigned_to: str

class TaskStatusUpdate(BaseModel):
    status: Literal["pending", "in_progress", "completed"]

class ReportResponse(BaseModel):
    id: int
    text: str
    latitude: float
    longitude: float
    image_url: Optional[str] = None
