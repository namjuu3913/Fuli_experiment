import json
from pydantic import BaseModel
# json output

# base
class UserOutBase(BaseModel):
    # Normal = no error, Error = error
    is_normal:bool

# error response
class UserOut_error(UserOutBase):
    is_normal:bool = False
    error_location: str
    error_detail: str 

# output of start LLM request(Item_startLLMServer)
class UserOut_startLLMResponse(UserOutBase):
    is_LLM_server_started: bool
    llm_server_info: dict
# output of stop_llm_server
class UserOut_stopLLMResponse(UserOutBase):
    is_normal:bool
    is_LLM_server_stopped:bool
    detail:str

# output of show_saved_character
class UserOut_showSavedCharacter(UserOutBase):
    Characters: list[dict]

# output of Select Character(Item_selecCharacter)
class UserOut_loadCharacter(UserOutBase):
    is_char_exists: bool
    loaded_character: dict

# output of generate character(Item_generateCharacter)
class UserOut_generateCharacter(UserOutBase):
    is_generated:bool

# output of requested chat (Item_Chat)
class UserOut_Chat(UserOutBase):
    character_name : str
    response : dict
    everything: dict

# output of server info request (Item_serverInfo)
class UserOut_serverInfo(UserOutBase):
    # python server info
    py_server_info: dict
    # llm server info
    llm_server_info: dict

# change psersonality (Item_changePersonality)
class UserOut_changePersonality(UserOutBase):
    # true: success     false: failed to change
    is_changed: bool