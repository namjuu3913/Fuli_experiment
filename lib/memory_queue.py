import json
from typing import Dict, List, Union, Deque
from collections import deque

class memory_queue:
    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.recent_memory: Deque[Dict] = deque()
    
    def add_memory(self, input_dict: Dict) -> Union[None, Dict]:
        removed_item: Union[None, Dict] = None
        
        if len(self.recent_memory) >= self.max_size:
            removed_item = self.recent_memory.popleft()
        
        self.recent_memory.append(input_dict)
        
        return removed_item

    def get_memory_json(self) -> str:
        memory_list = list(self.recent_memory)
        return json.dumps(memory_list, ensure_ascii=False, indent=2)
        
    
    def update_class(self, new_max:int):
        self.max_size = new_max