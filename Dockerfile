# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install git locally
RUN apt-get update && apt-get install -y git

# Install pip requirements & NLP dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY ./ ./

# Run FastAPI server
CMD ["uvicorn", "app.service.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
# uvicorn app.service.main:app --host 0.0.0.0 --port 8080 --reload


# Docker commands

# Build image
# docker build -t eda .
# docker rmi eda
# docker image ls

# Build/Remove container
# docker run --name eda-project-container -p 8080:8080 -d -v ${PWD}:/code data-analysis-image
# docker rm eda-project-container

# Show running containers
# docker ps  
# Show all containers
# docker ps -a

# Manage containers
# docker stop eda-project-container
# docker start eda-project-container