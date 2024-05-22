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
    AudioProcessor
)
import os
# import chromadb
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings
# )

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

"""
Starting vector store
"""

# vector_store_directory = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), "../../data/")
# )
# embedding_function = SentenceTransformerEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )
# vector_store = chromadb.PersistentClient(path=vector_store_directory)


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

    print(request)

    context = {
        "request": request,
        "user_message": message.content,
        "ai_message": chats[chat_id].send(message)
    }

    return templates.TemplateResponse("partials/message.html", context)


files_directory = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../data/")
)

from faster_whisper import WhisperModel

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    # news_processor = NewsProcessor(
    #     url="https://www.tagesschau.de/wirtschaft/google-suchmaschine-ki-chatgpt-100.html",
    #     vector_store=vector_store, 
    #     embedding_function=embedding_function
    # )

    # contents = await file.read()
    model = WhisperModel("tiny", cpu_threads=7, num_workers=4)
    segments, info = model.transcribe(audio=file.file)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    return JSONResponse(content={"filename": file.filename})
