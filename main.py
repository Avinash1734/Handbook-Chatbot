import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import jwt
from sqlalchemy.orm import Session

from . import models, database, vector_store, llm, utils

# Setup dirs
os.makedirs("uploads/pdfs", exist_ok=True)
os.makedirs("uploads/reports", exist_ok=True)

database.init_db()

app = FastAPI(title="Crisis Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks_db: Dict[int, models.Task] = {}
reports_db: Dict[int, models.ReportResponse] = {}
next_task_id = 1
next_report_id = 1

SECRET_KEY = "dev-key-only"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/auth/login", response_model=models.Token)
def login(req: models.LoginRequest):
    token = create_access_token({"sub": req.role})
    return {"access_token": token, "token_type": "bearer", "role": req.role}

@app.post("/standards/upload")
async def upload_standard(doc_name: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDFs allowed")
    temp_path = f"uploads/pdfs/{file.filename}"
    with open(temp_path, "wb") as buff:
        shutil.copyfileobj(file.file, buff)

    chunks = list(utils.chunk_pdf(temp_path, doc_name))
    if not chunks:
        raise HTTPException(400, "Failed to extract text")

    added = vector_store.add_chunks_to_db(doc_name, chunks)
    os.remove(temp_path)
    await file.close()
    return {"doc_name": doc_name, "chunks_added": added}

@app.post("/plans", response_model=models.PlanResponse)
def generate_plan(req: models.PlanRequest):
    q = f"Plan for {req.crisis_type} in {req.location} for {req.population} people. {req.constraints}"
    ctx = vector_store.query_standards(q, req.selected_standards, offline_mode=True)
    prompt = f"""{llm.get_system_prompt('manager')}

Act as a humanitarian planning assistant.

Crisis:
- Type: {req.crisis_type}
- Location: {req.location}
- Population: {req.population}
- Constraints: {req.constraints}

Standards context:
{ctx}

Respond with a plan.
"""
    wrapper = llm.LLMWrapper(mode="online")
    return wrapper.generate(prompt)

@app.post("/chat", response_model=models.ChatResponse)
def chat(req: models.ChatRequest, db: Session = Depends(database.get_db)):
    user_msg = database.ChatMessage(role="user", persona=req.role, message=req.message)
    db.add(user_msg); db.commit()

    prompt = f"{llm.get_system_prompt(req.role)}\n\nUser: {req.message}"
    wrapper = llm.LLMWrapper(mode=req.mode)
    resp = wrapper.generate(prompt)

    bot_msg = database.ChatMessage(role="assistant", persona=req.role, message=resp.answer)
    db.add(bot_msg); db.commit()
    return resp

# --- tasks ----------------------------------------------------------
@app.post("/tasks", response_model=models.Task, status_code=201)
def create_task(task: models.TaskCreate):
    global next_task_id
    t = models.Task(id=next_task_id, status="pending", **task.dict())
    tasks_db[next_task_id] = t
    next_task_id += 1
    return t

@app.get("/tasks", response_model=List[models.Task])
def get_tasks():
    return list(tasks_db.values())

@app.patch("/tasks/{task_id}/status", response_model=models.Task)
def update_task(task_id: int, status: models.TaskStatusUpdate):
    if task_id not in tasks_db:
        raise HTTPException(404, "task not found")
    tasks_db[task_id].status = status.status
    return tasks_db[task_id]

# --- reports --------------------------------------------------------
@app.post("/reports", response_model=models.ReportResponse, status_code=201)
async def create_report(
    text: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: Optional[UploadFile] = File(None),
):
    global next_report_id
    img_url = None
    if image:
        path = f"uploads/reports/{next_report_id}_{image.filename}"
        with open(path, "wb") as buff:
            shutil.copyfileobj(image.file, buff)
        img_url = f"/uploads/reports/{next_report_id}_{image.filename}"
        await image.close()

    r = models.ReportResponse(
        id=next_report_id, text=text, latitude=latitude, longitude=longitude, image_url=img_url
    )
    reports_db[next_report_id] = r
    next_report_id += 1
    return r

@app.get("/reports", response_model=List[models.ReportResponse])
def get_reports():
    return list(reports_db.values())

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.on_event("startup")
def startup():
    logging.basicConfig(level=logging.INFO)
    vector_store.get_standards_collection()
    logging.info("API ready.")
