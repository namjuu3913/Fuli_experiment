from pydantic import BaseModel
from typing import Optional

class Conversation(BaseModel):
    user_name:     Optional[str] = "Unknown User"
    user_context : str
    your_name:     Optional[str] = "You"
    your_context:  str