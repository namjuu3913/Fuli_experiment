# Fuli – Emotion-aware memory orchestrator

Fuli is the **agent-side brain** that connects:

- user input,
- semantic memories (FAISS + SentenceTransformer),
- the emotion engine (`deltaEGO`),
- and the LLM.

It currently uses a **vector-based RAG 1.0** architecture  
(FAISS semantic search over past conversations),
and is designed to be upgraded to **RAG 2.0 (graph-based)** in future work.

It does 3 main things:

1. Receives a VAD vector from the LLM and runs **deltaEGO** (search + analysis)  
2. Decides how “impressive” the current turn is and updates memories  
3. Retrieves relevant memories asynchronously and returns them as text context

---

## 🔧 Initialization

```python
from PythonServer.customPY.fuli import Fuli   # adjust path if needed

agent = Fuli(
    name="AllMight",
    short_mem_length=8,       # in-RAM queue (last N turns)
    recent_mem_length=32,     # FAISS recent memory capacity
    impressive_mem_length=32, # FAISS impressive memory capacity
    longterm_mem_length=0,    # (currently unused / TODO)
    background_length=0,      # (currently unused / TODO)
    emotion_load_num=5        # how many top emotions to store per turn
)
```
On initialization, Fuli:
  * loads existing FAISS indexes + JSON “general DB” from `CharacterSave/<name>/Memories/...`
  * creates empty indexes if files are missing
  * prepares a `deltaEGO` instance (`self.Carman`) and VAD config (`self.Ayin`)
---
## 🧠 VAD configuration (`Sephirah`, `Hod`, `Yesod`)
Fuli wraps VAD search/analysis configs into small dataclasses:
```python
@dataclass
class Hod:
    # search config for deltaEGO_VDB
    k: int = 5
    d: float | None = 0.3
    SIGMA: float = 0.6
    opt: str = "knn~gauss_w -B"

@dataclass
class Yesod:
    # analysis config for deltaEGO
    weights: weight = field(default_factory=lambda: copy.deepcopy(DEFAULT_WEIGHTS))
    var: variable = field(default_factory=lambda: copy.deepcopy(DEFAULT_VAR))
    ego_axis: EGO_axis = field(default_factory=lambda: copy.deepcopy(DEFAULT_EGO_AXIS))

@dataclass
class Sephirah:
    search_config: Hod = field(default_factory=Hod)
    analysis_config: Yesod = field(default_factory=Yesod)
```
Fuli keeps one instance:
```python
self.Ayin: Sephirah = Sephirah()
```
---
## 🧱 Memory system
**1) Short-term queue (`memory_queue`)**
```python
self.last_n_mem = memory_queue(short_mem_length)
```
  * Keeps the **last N conversations in RAM**.
  * Implemented as a `deque` with `maxlen`.
  * `add_memory()` returns the oldest item when it overflows:
    * Fuli then sends that popped memory into FAISS (recent / impressive DB).
      
**2) Vector DBs (FAISS)**
Fuli uses four logical memory types (two are currently active):
  * `recent_mem` (active)
  * `impressive_mem` (active)
  * `long_term_mem` (TODO)
  * `background` (TODO)

Each has:
  * a FAISS `IndexIDMap` (`*_mem_vec`)
  * a list of Pydantic models (`*_mem_gen`)

On startup:
```python
self.load_all_db()
```
loads or creates:
  * `.../recent_<name>_VDB.index` + `recent_<name>_GDB.json`
  * `../impressive_<name>_VDB.index` + `impressive_<name>_GDB.json`
  * (long-term/background are prepared but commented out for now)
    
All saves are done atomically with temp files to **be safe under multithreading**.
