from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

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
    llm: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0.2
        )
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
