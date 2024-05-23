from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory

from langchain_community.document_loaders import (
    WebBaseLoader,
    AsyncChromiumLoader
)
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough

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
        if they make a grammatical mistake. 
        If the user asks how you can help. Tell them, they can upload
        a website or podcast, and practice with its contents.
        They can for example get a random question asked to them using the button below.
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
    
    def is_query(self):
        return "@query" in self.content


class Chat(BaseModel):

    id: int
    messages: List[Message]
    llm: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0.2
        )
    )
    # llm: ChatOpenAI = Field(
    #     default_factory=lambda: ChatGroq(temperature=0.2, model="llama3-8b-8192")
    # )

    history: ChatMessageHistory = Field(
        default_factory=ChatMessageHistory
    )

    def send(self, message: Message, db: Chroma):
        
        if message.is_query():
            answer = Query(query=message.content, db=db).answer()
            
            self.history.add_user_message(message.content)
            self.history.add_ai_message(AIMessage(
                content = answer['answer'],
                response_metadata={
                    "sources": list(answer['sources'])
                }
            ))
            return {
                'question': message.content,
                'answer': answer['answer'],
                'sources': answer['sources']
            }
        else:
            chain = message.prompt() | self.llm
            ai_message = chain.invoke(
                {"messages": self.history.messages}
            )
            
            self.history.add_user_message(message.content)
            self.history.add_ai_message(ai_message.content)
            return {
                'question': message.content,
                'answer': ai_message.content,
                'sources': []
            }

from langchain_core.runnables import RunnableParallel

class Query:

    def __init__(self, query, db):
        self.query = query
        self.db = db
        self.retriever = db.as_retriever(search_type="mmr")  #, search_kwargs={"k": 5})
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self):
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self._format_docs(x["context"])))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        results = rag_chain_with_source.invoke(self.query)
        results['sources'] = list({
            document.metadata['source'] for document in results['context']
        })
        return results


class Ask:

    def __init__(self, docs, db):
        pass
    
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

    def _format_segments(self):
        def to_text(segment):
            return "[%.2fs -> %.2fs] %s" % (
                segment['start'],
                segment['end'],
                segment['text']
            )

        return "\n".join(to_text(seg) for seg in self.segments)

    def _generate_questions(self):

        logging.info("Formatting transcription")
        transcription = self._format_segments()

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        
        prompt_template = """
            The podcast transcription below was uploaded by a user.
            Generate questions about the main points of the podcast.
            Anwers these questions using only the transcription content.
            Make the answers long and include citations.
            Your questions and answers should be strictly in German.
            \n
            TRANSCRIPTION START
            \n
            {transcription}
            \n
            TRANSCRIPTION END
            \n      
            {format_instructions}
        """

        def prompt_from(format_instructions):
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["transcription"],
                partial_variables={
                    "format_instructions": format_instructions
                },
            )
            return prompt

        response_schemas=[
            ResponseSchema(
                name="quiz",
                description="""
                    description to model with the example: 
                    [{"question": "Question from content", "answer": "Answer using content"}]
                """,
                type="array(objects)"
            )
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        prompt = prompt_from(parser.get_format_instructions())

        chain = prompt | model | parser
        output = chain.invoke({"transcription": transcription})
        print(output)

    def save(self, db: Chroma):
        splits = []
        for segment in self.segments:
            splits.append(Document(
                page_content=segment['text'],
                metadata={
                    "start": segment['start'],
                    "end": segment['end']
                }
            ))

        # Generate questions from transcription with LLM and add to a Question collection
        self._generate_questions()

        logging.info("Indexing audio splits with VectorStore")
        db.from_documents(
            documents=splits,
            embedding=db.embeddings
        )
        return self._format_segments()



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

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def save(self, db: Chroma):
        logging.info("Indexing news article chunks with VectorStore")
        db.from_documents(
            documents=self.splits,
            embedding=db.embeddings
        )
        return self._format_docs(self.splits)


class TextProcessor:

    def __init__(self):
        pass