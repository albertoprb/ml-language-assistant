from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory

from langchain_community.document_loaders import WebBaseLoader
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough

import sqlite3

import whisperx

import logging
from langchain_core.runnables import RunnableParallel


class Message(BaseModel):
    content: str
    metadata: Optional[dict] = {}

    def prompt(self):
        sytem_message = """
        You are a helpful assistant. Your name is Sofia.
        You are a language teacher and this is a teaching session.
        You are only going to speak German. Never speak other language.
        Before you reply to the user, correct the user message if they made a grammatical mistake. Then continue with your reply.
        If they speak in english you can translate it for them and gently say, that's how you say it in German.
        Then proceed with your answer.
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

        return prompt

    def add_source(self, source):
        self.metadata["source"] = source

    def is_quiz(self):
        return "@quiz" in self.content

    def is_query(self):
        return "@query" in self.content


class Chat(BaseModel):
    id: int
    messages: List[Message]
    llm: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
    )

    history: ChatMessageHistory = Field(default_factory=ChatMessageHistory)

    def send(self, message: Message, vector_db_path, embedding_function):
        if message.is_query():
            answer = Query(
                query=message.content,
                vector_db_path=vector_db_path,
                embedding_function=embedding_function,
            ).answer()

            self.history.add_user_message(message.content)
            self.history.add_ai_message(
                AIMessage(
                    content=answer["answer"],
                    response_metadata={"sources": list(answer["sources"])},
                )
            )
            return {
                "question": message.content,
                "answer": answer["answer"],
                "sources": answer["sources"],
            }
        elif message.is_quiz():
            funny_quiz = "Guten Tag, Quizmaster! Los geht's!"
            self.history.add_user_message(funny_quiz)
            self.history.add_ai_message(
                AIMessage(
                    content=message.content,
                    response_metadata={"sources": [message.metadata["source"]]},
                )
            )
            return {
                "question": funny_quiz,
                "answer": message.content,
                "sources": [message.metadata["source"]],
            }
        else:
            chain = message.prompt() | self.llm
            ai_message = chain.invoke({"messages": self.history.messages})

            self.history.add_user_message(message.content)
            self.history.add_ai_message(ai_message.content)
            return {
                "question": message.content,
                "answer": ai_message.content,
                "sources": [],
            }


class Query:
    def __init__(self, query, vector_db_path, embedding_function):
        self.query = query
        self.db = vector_db_path
        chroma = Chroma(
            persist_directory=vector_db_path, embedding_function=embedding_function
        )
        self.retriever = chroma.as_retriever(
            search_type="mmr"
        )  # search_kwargs={"k": 5})
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self):
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self._format_docs(x["context"]))
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        results = rag_chain_with_source.invoke(self.query)

        results["sources"] = list(
            {  # Get unique sources
                document.metadata["source"] for document in results["context"]
            }
        )
        return results


class Quiz:
    prompt_template = """
        The CONTENT below was uploaded by a user.
        Generate questions about the main points of the CONTENT.
        Answer these questions using the CONTENT.
        The answers should be comprehensive and include CONTENT passages.
        Your questions and answers should be strictly in German.
        \n
        CONTENT START
        \n
        {context}
        \n
        CONTENT END
        \n     
        {format_instructions}
    """

    def __init__(self):
        self.qa = []

    def _prompt_from(self, format_instructions):
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context"],
            partial_variables={"format_instructions": format_instructions},
        )
        return prompt

    def questions_from(self, context):
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

        response_schemas = [
            ResponseSchema(
                name="quiz",
                description="""
                    description to model with the example:
                    [{
                        "question": "Question from content", 
                        "answer": "Answer using content and passages"
                    }]
                """,
                type="array(objects)",
            )
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        prompt = self._prompt_from(parser.get_format_instructions())

        chain = prompt | model | parser
        output = chain.invoke({"context": context})
        self.qa = output["quiz"]
        return self

    def save(self, db_path, source):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for qa in self.qa:
            cursor.execute(
                """
                INSERT INTO quiz (question, answer, source) 
                VALUES (?, ?, ?)
            """,
                (qa["question"], qa["answer"], source),
            )
        conn.commit()
        conn.close()

    @classmethod
    def random_question(cls, db_path):
        # The cls parameter refers to the class itself
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM quiz ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()

        if row is not None:
            columns = [column[0] for column in cursor.description]
            row_dict = dict(zip(columns, row))
            return row_dict
        else:
            return {"question": "Noch keine Fragen...", "answer": "", "source": ""}


class AudioProcessor:
    device = "cpu"
    batch_size = 16
    compute_type = "int8"
    model = "tiny"
    language = "de"
    segments: List = []
    quiz: Quiz = None

    def __init__(self, file, source, align=False):
        logging.info("Transcribing audio file %s", file)
        self.source = source
        model = whisperx.load_model(
            self.model,
            self.device,
            language=self.language,
            compute_type=self.compute_type,
        )

        audio = whisperx.load_audio(file=file)
        output = model.transcribe(audio=audio, batch_size=self.batch_size)

        if align:
            self.segments = self.align(file, output)["segments"]
        else:
            self.segments = output["segments"]

    def align(self, file, transcription_output):
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription_output["language"], device=self.device
        )

        output = whisperx.align(
            transcription_output["segments"],
            model_a,
            metadata,
            file,
            self.device,
            return_char_alignments=False,
        )

        return output

    def _format_segments(self):
        def to_text(segment):
            return "[%.2fs -> %.2fs] %s" % (
                segment["start"],
                segment["end"],
                segment["text"],
            )

        return "\n".join(to_text(seg) for seg in self.segments)

    def generate_questions(self):
        transcription = self._format_segments()
        self.quiz = Quiz().questions_from(transcription)
        return self.quiz.qa

    def save(self, vector_db_path, embedding_function, sql_db_path):
        splits = []
        for segment in self.segments:
            splits.append(
                Document(
                    page_content=segment["text"],
                    metadata={
                        "start": segment["start"],
                        "end": segment["end"],
                        "source": self.source,
                    },
                )
            )

        logging.info("Indexing audio splits with VectorStore")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=vector_db_path,
        )
        self.quiz.save(db_path=sql_db_path, source=self.source)
        return self._format_segments()


class NewsProcessor:
    quiz: Quiz = None

    def __init__(self, source):
        self.source = source
        logging.info("Reading the news from %s", source)
        loader = WebBaseLoader(
            web_paths=[source],
            bs_kwargs=dict(parse_only=bs4.SoupStrainer("article")),
        )
        docs = loader.load()

        logging.info("Splitting news article into chunks of 500 characters")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.splits = text_splitter.split_documents(docs)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_questions(self):
        context = self._format_docs(self.splits)
        self.quiz = Quiz().questions_from(context)
        return self.quiz.qa

    def save(self, vector_db_path, embedding_function, sql_db_path):
        logging.info("Indexing news article chunks with VectorStore")
        Chroma.from_documents(
            documents=self.splits,
            embedding=embedding_function,
            persist_directory=vector_db_path,
        )
        self.quiz.save(db_path=sql_db_path, source=self.source)
        return self._format_docs(self.splits)


class TextProcessor:
    def __init__(self):
        pass
