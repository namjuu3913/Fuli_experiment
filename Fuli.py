import numpy as np
import faiss, json, random
from pathlib import Path
from sentence_transformers import SentenceTransformer as ST
import asyncio
from typing import Union


ST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"/"snapshots"/"4328cf26390c98c5e3c738b4460a05b95f4911f5"
if not ST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find model folder: {ST_MODEL_PATH}")
model = ST(str(ST_MODEL_PATH), device='cpu')
EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()

class Fuli:
    def __init__(self, name: str):
        # is it initialized?
        self.flag : bool = False
        # demension
        self.d: int = EMBEDDING_DIMENSION
        # name of character
        self.name: str = name
        # counter flag(is over then 10?)
        self.cnt: int = 0; 

        # DB
        # recent_mem
        self.recent_mem_vec: faiss.Index
        self.recent_mem_gen: list[dict]
        # impressive_mem
        self.impress_mem_vec: faiss.Index
        self.impress_mem_gen: list[dict]
        # long_term_mem
        self.long_term_mem_vec: faiss.Index
        self.long_term_mem_gen: list[dict]
        # background
        self.background_vec: faiss.Index
        self.background_gen: list[dict]

        # PATH
        # recent memory
        self.recent_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"recent_mem"
        # impressive memory
        self.impressive_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"impressive_mem"
        # long term memory
        self.long_term_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"long_term_mem"
        # background
        self.background_path= Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"background"

        #load db
        try:
            self.load_all_db()
            self.flag = True
        except Exception as e:
            raise e.add_note("Failed to load db")
        
    def load_all_db(self):
        vector_path: Path
        general_path: Path
        
        # recent_mem
        vector_path = self.recent_mem_path / "INDEX" / f"recent_{self.name}_VDB.index"
        general_path = self.recent_mem_path / "SQL" / f"recent_{self.name}_GDB.json"
        self.load_db(vector_path, general_path, "recent") 

        # impressive_mem
        vector_path = self.impressive_mem_path / "INDEX" / f"impressive_{self.name}_VDB.index"
        general_path = self.impressive_mem_path / "SQL" / f"impressive_{self.name}_GDB.json"
        self.load_db(vector_path, general_path, "impressive")

        # long_term_mem
        vector_path = self.long_term_mem_path / "INDEX" / f"long_term_{self.name}_VDB.index"
        general_path = self.long_term_mem_path / "SQL" / f"long_term_{self.name}_GDB.json"
        self.load_db(vector_path, general_path, "long_term")

        # background
        vector_path = self.background_path / "INDEX" / f"background_{self.name}_VDB.index"
        general_path = self.background_path / "SQL" / f"background_{self.name}_GDB.json"
        self.load_db(vector_path, general_path, "background")

    def load_db(self, vec_path: Path, gen_path: Path, mem_type: str):
    
        mem_attributes = {
            "recent": ("recent_mem_gen", "recent_mem_vec"),
            "impressive": ("impress_mem_gen", "impress_mem_vec"),
            "long_term": ("long_term_mem_gen", "long_term_mem_vec"),
            "background": ("background_gen", "background_vec")
        }

        # Get proper attribute names
        attr_pair = mem_attributes.get(mem_type)
        if not attr_pair:
            print(f"Error: Unknown memory type: '{mem_type}'")
            return
        
        gen_attr_name, vec_attr_name = attr_pair

        # Do the files exist?
        if vec_path.exists() and gen_path.exists():
            print(f"'{mem_type}' DB loading............")
            
            # Load JSON (General DB)
            try:
                with open(gen_path, 'r', encoding='utf-8') as f:
                    setattr(self, gen_attr_name, json.load(f))
            except json.JSONDecodeError:
                print(f"Warning! : The file {gen_path} is corrupted. Initializing to a blank list....")
                setattr(self, gen_attr_name, [])
            
            # Load Faiss (Vector DB)
            try:
                setattr(self, vec_attr_name, faiss.read_index(str(vec_path)))
            except Exception as e:
                print(f"Warning! : Failed to load Faiss index {vec_path}. Creating a new one... Error: {e}")
                # In RAG, ID mapping is essential, so we use IndexIDMap.
                index = faiss.IndexIDMap(faiss.IndexFlatL2(self.d))
                setattr(self, vec_attr_name, index)

        else:
            # If the files do not exist
            print(f"'{mem_type}' DB file not found. Creating new ones.")
            
            # Initialize General DB (GDB) with an empty list
            setattr(self, gen_attr_name, [])
            
            # Initialize Vector DB (VDB) with an empty Faiss index
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.d))
            setattr(self, vec_attr_name, index)
            
            # Pre-create the folders for saving
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            gen_path.parent.mkdir(parents=True, exist_ok=True)

    # embedding the text(for now, it's 768 demension)
    @staticmethod
    def _get_vector_from_text(text: str) -> np.ndarray:
        embedding = model.encode(text)
        # np.atleast_2d: translate from (768,) to (1, 768)
        embedding_2d = np.atleast_2d(embedding)
        return embedding_2d.astype(np.float32)
    
    def add_conv_as_memory(self, conversation: dict) -> None:
        # embedding the user text
        vectored_text = Fuli._get_vector_from_text(conversation['context']['user'])

        # --- save ---
        # get impressive abs value of VAD vector -> abs(impressive_Vector)
        impressiveness: Union[bool, int] = self.get_impressive(conversation['emotion'])
        # does it succed to make it as vector?
        if isinstance(impressiveness, bool):
            raise Exception("Failed to make VAD vector value")
        # if value is more then 1
        if isinstance(impressiveness, int) and impressiveness > 100:
            raise Exception("Absolute value of VAD vector exceeded 100")
        # add to dict
        conversation["impressiveness"] = impressiveness
        
        # is it impressive?
        if impressiveness > 70:
            new_id = len(self.impress_mem_gen)
            faiss_id = np.array([new_id], dtype='int64')
            self.impress_mem_vec.add_with_ids(vectored_text, faiss_id)
            self.impress_mem_gen.append(conversation)
            self.cnt += 1
        else:
            new_id = len(self.recent_mem_gen)
            faiss_id = np.array([new_id], dtype='int64')
            self.recent_mem_vec.add_with_ids(vectored_text, faiss_id)
            self.recent_mem_gen.append(conversation)
            self.cnt += 1
        
        if(self.cnt >= 10):
            try:
                self.save_recent_and_impress()
                self.cnt = 0
            except:
                raise Exception("Failed to save last 10 conversation to db")
            
    def get_impressive(self, VAD:dict) -> Union[bool, int]:
        # get VAD vector values
        try:
            V:float = VAD['Valence']
            A:float = VAD['Arousal']
            D:float = VAD['Dominance']
        except:
            return False
        # are VAD values' ranges -1 ~ 1?
        if not all(-1.0 <= val <= 1.0 for val in [V, A, D]):
            print(f"Warning: VAD values exceeded range(-1 ~ 1). {VAD}")
            return False
        # calculate vector
        magnitude = np.linalg.norm([V, A, D])
        max_magnitude = np.sqrt(3)
        normalized_score = magnitude / max_magnitude
        impressiveness_score = int(normalized_score * 100)
        return impressiveness_score
        
    def save_recent_and_impress(self):
        print("Saving all DBs to disk...")
        print("Saving all DBs to disk...")
        
        # 1. recent_mem
        vec_path = self.recent_mem_path / "INDEX" / f"recent_{self.name}_VDB.index"
        gen_path = self.recent_mem_path / "SQL" / f"recent_{self.name}_GDB.json"       
        # Faiss save index
        faiss.write_index(self.recent_mem_vec, str(vec_path))
        # JSON save list
        with open(gen_path, 'w', encoding='utf-8') as f:
            json.dump(self.recent_mem_gen, f, indent=2, ensure_ascii=False)

        # 2. impressive_mem save
        vec_path = self.impressive_mem_path / "INDEX" / f"impressive_{self.name}_VDB.index"
        gen_path = self.impressive_mem_path / "SQL" / f"impressive_{self.name}_GDB.json"
        # Faiss save index
        faiss.write_index(self.impress_mem_vec, str(vec_path))
        # JSON save list
        with open(gen_path, 'w', encoding='utf-8') as f:
            json.dump(self.impress_mem_gen, f, indent=2, ensure_ascii=False)
        
