from typing import Optional
from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates  # For HTML templates
from fastapi.staticfiles import StaticFiles  # for mounting static files

import os  # For file paths
import debugpy  # For debugging

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


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    hx_request: Optional[str] = Header(None)
):

    context = {
        "request": request
    }

    return templates.TemplateResponse("index.html", context)
