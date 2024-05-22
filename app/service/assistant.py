from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

from langchain_community.document_loaders import (
    WebBaseLoader,
    AsyncChromiumLoader
)
import bs4

from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from faster_whisper import WhisperModel

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

    def __init__(self, file):
        model = WhisperModel("tiny", cpu_threads=7, num_workers=4)
        segments, info = model.transcribe(file)
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


class NewsProcessor:

    def __init__(self, url, vector_store, embedding_function):
        
        logging.info("Loading news article from %s", url)
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer('article')
            ),
        )
        docs = loader.load()

        logging.info("Splitting news article into chunks of 1000 characters")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)

        logging.info("Indexing news article chunks with VectorStore")
        chroma = Chroma.from_documents(
            client=vector_store,
            documents=splits,
            embedding=embedding_function
        )
        
        query = "Was Plant Google?"
        results = chroma.similarity_search(query)

        # print results
        print(results)


class TextProcessor:

    def __init__(self):
        pass
