from custom_lib.deltaEGO import InstantMetrics, DynamicMetrics, CumulativeMetrics, AnalysisResult_py
from pydantic import BaseModel, Field
from typing import Optional
from default_class import Conversation


class VADModel(BaseModel):
    V: float = Field(..., ge=-1.0, le=1.0, alias="Valence")     
    A: float = Field(..., ge=-1.0, le=1.0, alias="Arousal")
    D: float = Field(..., ge=-1.0, le=1.0, alias="Dominance")

class tokens(BaseModel):
    stress :        float
    reward :        float
    shocking_level :float

class general_mem(BaseModel):
    emotion         : dict[str]
    state_tokens    : tokens
    impressiveness  : int   
    context         : Conversation
    time_stamp      : str

class long_term_mem(BaseModel):
    title      : str
    time_stamp : str
    diary      : str

class Fuli_LOG(BaseModel):
    character_mem   : general_mem
    VAD             : VADModel
    analysis        : AnalysisResult_py
    search_log      : dict
    time_stamp      : str
