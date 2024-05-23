from typing import Optional
from fastapi import FastAPI, Request, Header, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates  # For HTML templates
from fastapi.staticfiles import StaticFiles  # for mounting static files
import logging
import debugpy  # For debugging
from app.service.assistant import (
    Chat,
    Message,
    NewsProcessor,
    AudioProcessor,
    Query
)
import os
import shutil
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)

import whisperx
# from faster_whisper import WhisperModel

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

"""
Starting vector store
"""

vector_store_directory = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../data/")
)
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vector_store = chromadb.PersistentClient(path=vector_store_directory)
langchain_chroma = Chroma(
    client=vector_store,
    embedding_function=embedding_function,
)

"""
Starting app
"""

# For debugging
debugpy.listen(("0.0.0.0", 5678))

# Create the FastAPI app
app = FastAPI()


"""
Loading static assets
"""

# Set-up templating engine
templates_directory = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../templates/")
)
templates = Jinja2Templates(directory=templates_directory)

# Mount static assets
app.mount(
    "/assets",
    StaticFiles(directory="./app/assets"),
    name="assets"
)
app.mount(
    "/preline",
    StaticFiles(directory="./app/node_modules/preline"),
    name="preline"
)

"""
Routes
"""

# Chats memory
chats = {0: Chat(id=0, messages=[])}

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    hx_request: Optional[str] = Header(None)
):

    chat = chats[0]

    context = {
        "request": request,
        "chat": chat
    }

    return templates.TemplateResponse("index.html", context)


@app.post("/chats/{chat_id}/messages/")
async def send_message(
    request: Request,
    message: Message,
    chat_id: int
):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")

    reply = chats[chat_id].send(
        message=message,
        db=langchain_chroma
    )
    logging.info("Reply: %s", reply)
    context = {
        "request": request,
        "user_message": reply["question"],
        "ai_message": reply["answer"],
        "sources": reply["sources"],
    }

    return templates.TemplateResponse("partials/message.html", context)


@app.post("/content/audio/")
async def audio(file: UploadFile = File(...)):

    if not file:
        return JSONResponse(content={"error": "No file sent"}, status_code=400)
    else:
        file_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "../../data/" + 
                file.filename
            )
        )
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        processed_audio = AudioProcessor(file=file_path)
        segments = processed_audio.save(db=langchain_chroma)
        return JSONResponse(
            content={
                "filename": file.filename,
                "segments": segments
            }, 
            status_code=200
        )

@app.post("/content/news/")
async def news(request: Request):

    data = await request.json()
    url = data.get("url")

    if not url:
        return JSONResponse(content={"error": "No url sent"}, status_code=400)
    else:
        news_processor = NewsProcessor(url=url)
        return JSONResponse(
            content={
                "url": url,
                "text": news_processor.save(db=langchain_chroma)
            }, 
            status_code=200
        )

@app.get("/query/")
async def find(query: str):

    if query == "":
        return JSONResponse(content={"error": "Empty query"}, status_code=400)
    else:
        query = Query(query=query, db=langchain_chroma)
        return JSONResponse(content={"answer": query.answer()}, status_code=200)
