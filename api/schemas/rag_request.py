from pydantic import BaseModel


class PublicChatRequest(BaseModel):
    question: str