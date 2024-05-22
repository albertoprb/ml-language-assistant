from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory

from langchain_community.document_loaders import (
    WebBaseLoader,
    AsyncChromiumLoader
)
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import whisperx

import logging


class Message(BaseModel):
    content: str

    def prompt(self):

        sytem_message = """
        You are a helpful assistant named Sofia.
        You are a language teacher. This is a teaching session.
        You are only going to speak German.
        Before you reply to the user, correct the user message \
        there's a grammatical mistake. Then proceed with your answer.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    sytem_message,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        print("##### prompt", prompt)

        return prompt


class Chat(BaseModel):

    id: int
    messages: List[Message]
    # llm: ChatOpenAI = Field(
    #     default_factory=lambda: ChatOpenAI(
    #         model="gpt-3.5-turbo-1106",
    #         temperature=0.2
    #     )
    # )
    llm: ChatOpenAI = Field(
        default_factory=lambda: ChatGroq(temperature=0.2, model="llama3-8b-8192")
    )

    history: ChatMessageHistory = Field(
        default_factory=ChatMessageHistory
    )

    def send(self, message: Message):
        chain = message.prompt() | self.llm

        self.history.add_user_message(message.content)

        ai_message = chain.invoke(
            {"messages": self.history.messages}
        )

        self.history.add_ai_message(ai_message.content)

        return ai_message.content


class AudioProcessor:

    device = "cpu"
    batch_size = 16
    compute_type = "int8"
    model = "tiny"
    language = "de"
    segments: List = []

    def __init__(self, file, align=False):
        logging.info("Transcribing audio file %s", file)
        model = whisperx.load_model(
            self.model,
            self.device,
            language=self.language,
            compute_type=self.compute_type
        )

        audio = whisperx.load_audio(file=file)
        output = model.transcribe(audio=audio, batch_size=self.batch_size)

        if align:
            self.segments = self.align(file, output)['segments']
        else:
            self.segments = output["segments"]

    def align(self, file, transcription_output):
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription_output["language"],
            device=self.device
        )

        output = whisperx.align(
            transcription_output["segments"],
            model_a,
            metadata,
            file,
            self.device,
            return_char_alignments=False
        )

        return output

    def save(self, vector_store, embedding_function):
        splits = []
        for segment in self.segments:
            splits.append(Document(
                page_content=segment['text'],
                metadata={
                    "start": segment['start'],
                    "end": segment['end']
                }
            ))

        logging.info("Indexing audio splits with VectorStore")
        Chroma.from_documents(
            client=vector_store,
            documents=splits,
            embedding=embedding_function
        )

        return splits


class NewsProcessor:

    def __init__(self, url):

        logging.info("Reading the news from %s", url)
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer('article')
            ),
        )
        docs = loader.load()

        logging.info("Splitting news article into chunks of 500 characters")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0
        )
        self.splits = text_splitter.split_documents(docs)

    def save(self, vector_store, embedding_function):
        logging.info("Indexing news article chunks with VectorStore")
        Chroma.from_documents(
            client=vector_store,
            documents=self.splits,
            embedding=embedding_function
        )


class TextProcessor:

    def __init__(self):
        pass
