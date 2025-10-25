import numpy as np
import faiss, json, random
from pathlib import Path
from sentence_transformers import SentenceTransformer as ST
import asyncio

ST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"/"snapshots"/"4328cf26390c98c5e3c738b4460a05b95f4911f5"
if not ST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find model folder: {ST_MODEL_PATH}")
model = ST(str(ST_MODEL_PATH), device='cpu')
EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()

class Fuli:
    # is it initialized?
    flag : bool = False
    # demension
    d: int
    # name of character
    name: str

    # PATH
    recent_mem_path: Path
    impressive_mem_path: Path
    long_term_mem_path: Path
    background_path: Path

    # DB
    # recent_mem
    recent_mem_vec: faiss.Index
    recent_mem_gen: list[dict]
    # impressive_mem
    impress_mem_vec: faiss.Index
    impress_mem_gen: list[dict]
    # long_term_mem
    long_term_mem_vec: faiss.Index
    long_term_mem_gen: list[dict]
    # background
    background_vec: faiss.Index
    background_gen: list[dict]


    def __init__(self, name: str):
        self.name = name
        self.flag = False

        self.d = EMBEDDING_DIMENSION

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
        # save
        if(conversation):
            asdf