import numpy as np
import faiss, json, random
from pathlib import Path
from sentence_transformers import SentenceTransformer as ST

ST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"/"snapshots"/"4328cf26390c98c5e3c738b4460a05b95f4911f5"
if not ST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find model folder: {ST_MODEL_PATH}")
model = ST(str(ST_MODEL_PATH), device='cpu')
EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()

class Fuli:
    def __init__(self, name: str):
        self.name = name
        # vector db and sql db path
        self.memory_path_sql = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"memory"/"SQL"
        self.memory_path_index = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"memory"/"INDEX"
        self.background_path_sql = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"background"/"SQL"
        self.background_path_index = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"background"/"INDEX"