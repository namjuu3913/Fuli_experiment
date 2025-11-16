from custom_lib.deltaEGO import deltaEGO,  weight, variable, EGO_axis, AnalysisResult_py
from custom_lib.memory import VADModel, tokens, general_mem, long_term_mem, Fuli_LOG
from sentence_transformers import SentenceTransformer as ST
from pydantic import ValidationError
from typing import List, Union, Deque
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Union
from default_class import Conversation
from datetime import datetime
import faiss, json, os, tempfile, copy
import asyncio, threading
import numpy as np
import warnings

ST_MODEL_PATH = Path(__file__).resolve().parent / "models" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"/"snapshots"/"4328cf26390c98c5e3c738b4460a05b95f4911f5"
if not ST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find model folder: {ST_MODEL_PATH}")
model = ST(str(ST_MODEL_PATH), device='cpu')
EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()

# for memory -------------------------------------------------------------
class memory_queue:
    def __init__(self, max_size: int):
        self.recent_memory: Deque[general_mem] = deque(maxlen=max_size)

    def add_memory(self, input_model: general_mem) -> Union[None, general_mem]:
        removed_item: Union[None, general_mem] = None
        
        if len(self.recent_memory) == self.recent_memory.maxlen:
            removed_item = self.recent_memory[0] 
        
        self.recent_memory.append(input_model)
        
        return removed_item

    def get_memory_json(self) -> str:
        memory_list = list(self.recent_memory)
        memory_list_as_dicts = [mem.model_dump() for mem in memory_list]
        
        return json.dumps(memory_list_as_dicts, ensure_ascii=False, indent=2)
    
    def update_class(self, new_max: int):
        current_items = list(self.recent_memory)
        self.recent_memory = deque(current_items, maxlen=new_max)
           
    def __len__(self) -> int:
        return len(self.recent_memory)

    def __iter__(self):
        return iter(self.recent_memory)
# for memory -------------------------------------------------------------

# VAD ---------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "weightA_stress" : 0.1,
    "weightV_stress" : 0.9,
    "weightA_reward" : 0.7,
    "weightV_reward" : 0.3,
    "weight_k"       : 0.5
}

DEFAULT_VAR = {
    "theta_0" : 0,
    "dampening_factor" : 0.08
}

DEFAULT_EGO_AXIS = {
    "baseline" : {
        "v" : 0.0,
        "a" : 0.0,
        "d" : 0.0,
        "owner" : "",
        "timestamp" : 0.0
    },
    "stabilityRadius" : 0.55
}
@dataclass
class Hod:
    """A dataclass that has search config of Sephirah"""
    k:      int = 5
    d:      Union[int, None] = 0.3
    SIGMA:  float = 0.6
    opt:    str = "knn~gauss_w -B"
@dataclass
class Yesod:
    """A dataclass that has analysis config of Sephirah""" 
    weights: weight = field(default_factory=lambda: copy.deepcopy(DEFAULT_WEIGHTS))
    var: variable = field(default_factory=lambda: copy.deepcopy(DEFAULT_VAR))
    ego_axis: EGO_axis = field(default_factory=lambda: copy.deepcopy(DEFAULT_EGO_AXIS))
@dataclass
class Sephirah:
    """A dataclass that has config for VAD search and analysis"""
    search_config   :Hod = field(default_factory = Hod)
    analysis_config :Yesod = field(default_factory = Yesod)
# VAD ---------------------------------------------------------------------

# parsed AI response ------------------------------------------------------
class LLM_output:
    context:    Conversation
    VAD:        VADModel  
# parsed AI response ------------------------------------------------------


class Fuli:
    def __init__(self, name: str, short_mem_length: int, 
                 recent_mem_length:int, impressive_mem_length:int, 
                 longterm_mem_length:int, background_length:int, emotion_load_num:int):
        # is it initialized?
        self.flag : bool = False
        # demension
        self.d: int = EMBEDDING_DIMENSION
        # name of character
        self.name: str = name
        # how many emotion will be loaded?
        self.emotion_num = emotion_load_num 
        # counter flag(is over then 10?)
        self.cnt: int = 0; 

        # last 8 conversations
        self.last_n_mem:memory_queue = memory_queue(short_mem_length)
        # how many recent mem?
        self.recent_mem_num:int = recent_mem_length
        self.impressive_mem_num:int = impressive_mem_length

        # DB
        # recent_mem
        self.recent_mem_vec: faiss.Index
        self.recent_mem_gen: list[general_mem]
        # impressive_mem
        self.impress_mem_vec: faiss.Index
        self.impress_mem_gen: list[general_mem]
        # long_term_mem
        self.long_term_mem_vec: faiss.Index
        self.long_term_mem_gen: list[long_term_mem]
        # background
        self.background_vec: faiss.Index
        self.background_gen: list[dict]
        # LOG
        self.LOG: list[Fuli_LOG] = []

        # PATH
        # recent memory
        self.recent_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"recent_mem"
        # impressive memory
        self.impressive_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"impressive_mem"
        # long term memory
        # self.long_term_mem_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"long_term_mem"
        # background
        # self.background_path= Path(__file__).resolve().parent / "CharacterSave" / self.name/"Memories"/"background"
        # LOG
        self.LOG_path = Path(__file__).resolve().parent / "CharacterSave" / self.name/ "LOG"

        self._locks = {
            "recent": threading.RLock(),
            "impressive": threading.RLock(),
            # add from here
        }

        # VAD related---------------------------- 
        # VAD emotion module (PM name)
        self.Carman = deltaEGO(self.name)

        # VAD saearch and analysis variables (PM name)
        self.Ayin: Sephirah = Sephirah()

        # VAD variables
        self.VAD_search_result: Union[dict, None] = None
        self.VAD_analysis_result: Union[AnalysisResult_py, None] = None
        self.simple_emotion_result:Union[List[str], None] = None
        self.simple_emotion_analysis_token: Union[tokens, None] = None

        # did it searched emotion?
        self.Abnomality: bool = False       
        # VAD related---------------------------- 

        #load db
        try:
            self.load_all_db()
            self.flag = True
        except Exception as e:
            raise e.add_note("Failed to load db")
        

    # embedding the text(for now, it's 768 demension)
    @staticmethod
    def _get_vector_from_text(text: str) -> np.ndarray:
        embedding = model.encode(text)
        # np.atleast_2d: translate from (768,) to (1, 768)
        embedding_2d = np.atleast_2d(embedding)
        return embedding_2d.astype(np.float32)
        
# load DBs---------------------------------------------------------------------------------------------------       
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
        # vector_path = self.long_term_mem_path / "INDEX" / f"long_term_{self.name}_VDB.index"
        # general_path = self.long_term_mem_path / "SQL" / f"long_term_{self.name}_GDB.json"
        # self.load_db(vector_path, general_path, "long_term")

        # background
        #vector_path = self.background_path / "INDEX" / f"background_{self.name}_VDB.index"
        #general_path = self.background_path / "SQL" / f"background_{self.name}_GDB.json"
        #self.load_db(vector_path, general_path, "background")


    def load_db(self, vec_path: Path, gen_path: Path, mem_type: str):
    
        # A dictionary that mapping Pydantic model
        model_map = {
            "recent": general_mem,
            "impressive": general_mem,
            "long_term": long_term_mem,
            "background": dict
        }
        ModelClass = model_map.get(mem_type)
        if not ModelClass:
            print(f"Error: Unknown memory type: '{mem_type}'")
            return
            
        mem_attributes = {
            "recent": ("recent_mem_gen", "recent_mem_vec"),
            "impressive": ("impress_mem_gen", "impress_mem_vec"),
            "long_term": ("long_term_mem_gen", "long_term_mem_vec"),
            "background": ("background_gen", "background_vec")
        }

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
                    data_list = json.load(f)  

                    if ModelClass != dict:
                        validated_data = [ModelClass.model_validate(item) for item in data_list]
                        setattr(self, gen_attr_name, validated_data)
                    else:
                        setattr(self, gen_attr_name, data_list)

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Warning! : The file {gen_path} is corrupted or invalid. {e}")
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
# load DBs---------------------------------------------------------------------------------------------------       

# input -----------------------------------------------------------------------------------------------------

# input emotion *first*--------------------------------------------------------------------------------------
    def get_emotion(self, VAD_str: str) -> bool:
        try:
            VAD = json.loads(VAD_str)
            V: float = VAD['Valence']
            A: float = VAD['Arousal']
            D: float = VAD['Dominance']
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"VAD parsing or key error! : {e}")
            return False
            
        # are VAD values' ranges -1 ~ 1?
        if not all(-1.0 <= val <= 1.0 for val in [V, A, D]):
            print(f"Warning: VAD values exceeded range(-1 ~ 1). {VAD}")
            return False
        
        # search
        try:
            self.VAD_search_result = self.Carman.VADsearch(in_VAD = {
                "V"  : V, "A"  : A, "D"  : D, 
                "dis"   : self.Ayin.search_config.d,
                "k"     : self.Ayin.search_config.k,
                "sigma" : self.Ayin.search_config.SIGMA,
                "api"   : self.Ayin.search_config.opt
                })
        except Exception as e:
            print(f"Search Fail! Location:{Path(__file__)}. Error: {e}")
            return False
        # analysis
        try:
            self.VAD_analysis_result = self.Carman.analize_VAD(
                weights = self.Ayin.analysis_config.weights,
                variables = self.Ayin.analysis_config.var,
                emotion_base = self.Ayin.analysis_config.ego_axis,
                return_analysis = True,
                append_emotion = True
            )
        except Exception as e:
            print(f"Analysis Fail! Location:{Path(__file__)}. Error: {e}")
            return False
        
        parsed_data = self.VAD_search_result
        emotion_result_array = parsed_data.get("result", [])
        self.simple_emotion_result = [item['term'] for item in emotion_result_array[:self.emotion_num] if 'term' in item]

        # get state tokens
        self.simple_emotion_analysis_token = tokens(
            stress = self.VAD_analysis_result['instant']['stress_ratio'],
            reward = self.VAD_analysis_result['instant']['reward_ratio'],
            shockingLevel = self.VAD_analysis_result['dynamics']['affective_lability']
        )

        self.Abnomality = True
        return True
# input emotion *first*--------------------------------------------------------------------------------------

# input actual memory *second*-------------------------------------------------------------------------------    
    def get_impressive(self, VAD:VADModel) -> int:       
        # calculate vector
        magnitude = np.linalg.norm([VAD.V, VAD.A, VAD.D])
        max_magnitude = np.sqrt(3)
        normalized_score = magnitude / max_magnitude
        impressiveness_score = int(normalized_score * 100)
        
        if impressiveness_score > 100:
            warnings.warn("Absolute value of VAD vector exceeded 100")
            return 100
        
        return impressiveness_score
    
    
    def update_memory(self, conversation: Conversation) -> None:
        if not self.Abnomality:
            raise Exception(f"Emotion is not searched! Fatal logic error!!!!!! CALL THE POLICE!!!!!!")
        current_vad = VADModel(
            V = self.VAD_search_result['query']['V'],
            A = self.VAD_search_result['query']['A'],
            D = self.VAD_search_result['query']['D']
        )
        time_stamp_now: str = datetime.now().isoformat()
        # update from here
        current_impressiveness: int = self.get_impressive(current_vad)
        mem: general_mem = self.make_new_gen_mem(conversation, current_impressiveness, time_stamp_now)

        popped_mem: Union[None, general_mem] = self.last_n_mem.add_memory(mem)    
        if isinstance(popped_mem, general_mem):
            self.add_conv_as_memory(popped_mem)

        new_LOG: Fuli_LOG = Fuli_LOG(
            character_mem = mem,
            VAD = current_vad,
            analysis = self.VAD_analysis_result,
            search_log = self.VAD_search_result,
            time_stamp = time_stamp_now
        )
        self.LOG.append(new_LOG)        
        self.VAD_analysis_result = None
        self.VAD_search_result = None
        self.simple_emotion_analysis_token = None
        self.simple_emotion_result = None
        self.Abnomality = False

#TODO asyncio
    def add_conv_as_memory(self, mem: general_mem) -> general_mem:
        # embedding the user text
        vectored_text = Fuli._get_vector_from_text(mem.context.user_context)    

        # is it impressive?
        if mem.impressiveness > 80:
             with self._locks["impressive"]:
                new_id = len(self.impress_mem_gen)
                self.impress_mem_vec.add_with_ids(vectored_text, np.array([new_id], dtype='int64'))
                self.impress_mem_gen.append(mem)
        else:
            with self._locks["recent"]:
                new_id = len(self.recent_mem_gen)
                self.recent_mem_vec.add_with_ids(vectored_text, np.array([new_id], dtype='int64'))
                self.recent_mem_gen.append(mem)
        
        self.cnt += 1
        
        if(self.cnt >= 10):
            try:
                self.save_recent_and_impress()
                self.cnt = 0
            except:
                raise Exception("Failed to save last 10 conversation to db")

    def make_new_gen_mem(self, context_in: Conversation, impressiveness_in: int, time_stamp_in:str) -> general_mem:
        # make new memory
        new_memory : general_mem = general_mem(
            emotion = self.simple_emotion_result,
            impressiveness = impressiveness_in,
            time_stamp = time_stamp_in,
            context = context_in,
            state_tokens = self.simple_emotion_analysis_token
        )

        return new_memory
# input actual memory *second*-------------------------------------------------------------------------------    
            
    def save_recent_and_impress(self) -> None:
        errs = []

        def safe_save(kind, vec_path, gen_path, vec_obj, gen_obj):
            lock = self._locks[kind]
            try:
                with lock:
                    self._atomic_write_faiss(vec_obj, vec_path)
                    self._atomic_write_json(gen_obj, gen_path)
            except Exception as e:
                errs.append((kind, e))

        # prepare path
        recent_vec_path = self.recent_mem_path / "INDEX" / f"recent_{self.name}_VDB.index"
        recent_gen_path = self.recent_mem_path / "SQL" / f"recent_{self.name}_GDB.json"
        impress_vec_path = self.impressive_mem_path / "INDEX" / f"impressive_{self.name}_VDB.index"
        impress_gen_path = self.impressive_mem_path / "SQL" / f"impressive_{self.name}_GDB.json"

        t1 = threading.Thread(target=safe_save, args=("recent", recent_vec_path, recent_gen_path, self.recent_mem_vec, self.recent_mem_gen))
        t2 = threading.Thread(target=safe_save, args=("impressive", impress_vec_path, impress_gen_path, self.impress_mem_vec, self.impress_mem_gen))
        t1.start(); t2.start(); t1.join(); t2.join()

        if errs:
            # returns only 1 error
            kind, e = errs[0]
            raise RuntimeError(f"{kind} save fail") from e
               
    # for safe multithreading    
    def _atomic_write_json(self, data, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = []
        if data and hasattr(data[0], 'model_dump'):
             data_to_save = [item.model_dump() for item in data]
        else:
             data_to_save = data # if it is dict or list
             
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
            # save converted dict(data_to_save)
            json.dump(data_to_save, tmp, ensure_ascii=False, indent=2) 
            tmp.flush(); os.fsync(tmp.fileno())
            tmp_name = tmp.name

        os.replace(tmp_name, path)

    def _atomic_write_faiss(self, index, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
            tmp_name = tmp.name
        faiss.write_index(index, tmp_name)

        os.replace(tmp_name, path)    


#----------------get mem---------------------------------------------------------------------------    
    async def get_memories(self, input:str) -> str:
        # this will be returned
        reval:str = ""
        # add last 8 mem to reval
        reval += "last 8 conversation: \n"
        if len(self.last_n_mem) == 0:
            reval += "(No recent conversations)\n"
        else:
            for mem in self.last_n_mem:
                reval += f"{str(mem)}\n"          

        # get vectored db
        # embed the input (it is using a lot of cpu)
        vectored_in = await asyncio.to_thread(self._get_vector_from_text, input)
        
        # search mamory with multithreading
        search_tasks = [
            asyncio.to_thread(self.search_recent_mem, vectored_in, self.recent_mem_num),
            asyncio.to_thread(self.search_impressive_mem, vectored_in, self.impressive_mem_num)
            #asyncio.to_thread(self.search_long_term_mem, vectored_in, 2)
            #asyncio.to_thread(self.search_background_mem, vectored_in, 5)
        ]

        # wait until it ends
        (recent_results, 
         impress_results
         #long_term_results 
         #background_results TODO
         ) = await asyncio.gather(*search_tasks)
        
        # add all results
        reval += "\n--- Relevant Recent Memories ---\n"
        if recent_results:
            for item in recent_results:
                reval += f"{str(item)}\n"
        else:
            reval += "(None)\n"

        reval += "\n--- Relevant Impressive Memories ---\n"
        if impress_results:
            for item in impress_results:
                reval += f"{str(item)}\n"
        else:
            reval += "(None)\n"

        #reval += "\n--- Relevant Long-Term Memories ---\n"
        #if long_term_results:
        #    for item in long_term_results:
        #        reval += f"{str(item)}\n"
        #else:
        #    reval += "(None)\n"

        #reval += "\n--- Relevant Background Information ---\n"
        #if background_results:
        #    for item in background_results:
        #        reval += f"{str(item)}\n"
        #else:
        #    reval += "(None)\n"
            
        return reval
       
    # thread functions
    # recent
    def search_recent_mem(self, query_vec: np.ndarray, k: int = 3) -> List[general_mem]:
        print("... (Thread) searching recent mem")
        try:
            return self._search_db(self.recent_mem_vec, self.recent_mem_gen, query_vec, k)
        except Exception as e:
            warnings.warn(f"Failed: Error from search_recent_mem Thread.\nDetails: {e}")
            return []          
    # impressive
    def search_impressive_mem(self, query_vec: np.ndarray, k: int = 2) -> List[general_mem]:
        print("... (Thread) searching impressive mem")
        try:
            return self._search_db(self.impress_mem_vec, self.impress_mem_gen, query_vec, k)
        except Exception as e:
            warnings.warn(f"Failed: Error from search_impressive_mem Thread.\nDetails: {e}")
            return []
        
    #-----------------------LATER-----------------------            
    # long_term TODO
    #def search_long_term_mem(self, query_vec: np.ndarray, k: int = 2) -> list[dict]:
    #    print("... (Thread) searching long-term mem")
    #    try:
    #        return self._search_db(self.long_term_mem_vec, self.long_term_mem_gen, query_vec, k)
    #    except Exception as e:
    #        print(f"Failed: Error from search_long_term_mem Thread.\nDetails: {e}")
    #        return [{"Error" : "Failed to load long-term memories"}]            
    # background TODO
    #def search_background_mem(self, query_vec: np.ndarray, k: int = 1) -> list[dict]:
    #    print("... (Thread) searching background mem")
    #    try:
    #        return self._search_db(self.background_vec, self.background_gen, query_vec, k)
    #    except Exception as e:
    #        print(f"Failed: Error from search_background_mem Thread.\nDetails: {e}")
    #        return [{"Error" : "Failed to load background memories"}]
    #-----------------------LATER-----------------------      

    # what will run in thread    
    def _search_db(self, vec_db: faiss.Index, gen_db: list[dict], query_vec: np.ndarray, k: int) -> list[dict]:
        if vec_db.ntotal == 0:
            return []

        actual_k = min(k, vec_db.ntotal)
        
        try:
            # D: Distance, I: Index(ID)
            D, I = vec_db.search(query_vec, actual_k)
            
            results = []
            for idx in I[0]:
                if idx == -1:
                    continue
                results.append(gen_db[int(idx)])
            return results
            
        except Exception as e:
            print(f"Error during DB search: {e}")
            return []
#--------------------------------------------------

    # when program truns off or delete this character, dump every log
    def turn_off(self) -> None:
        print(f"Turning off {self.name}. Saving session LOG...")
        
        # time stamp
        timestamp_now = datetime.now()
        timestamp_str = timestamp_now.strftime('%Y%m%d_%H%M%S') # 파일명에 적합한 형식
        
        # path
        file_name = f"{self.name}-{timestamp_str}.json"
        file_path = self.LOG_path / file_name
        
        self.LOG_path.mkdir(parents=True, exist_ok=True)
        
        try:
            data_to_save = [log.model_dump() for log in self.LOG]
        except Exception as e:
            print(f"Cannot save the log: {e}.")
            return

        # save it as json
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(self.LOG_path)) as tmp:
                json.dump(data_to_save, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            
            os.replace(tmp_name, file_path)
            print(f"LOG save compelete! : {file_path}")
            
        except Exception as e:
            print(f"!!! Failed to save log !!! : {file_path}. Error : {e}")
            if 'tmp_name' in locals() and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except:
                    pass
