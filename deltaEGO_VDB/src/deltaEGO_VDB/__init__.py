import importlib.resources
import json
from . import core

class EGOSearcher:
    def __init__(self):
        self._cpp_tree = core.KDTree()
        
        try:
            json_path_obj = importlib.resources.files('deltaEGO_VDB').joinpath('VAD.json')
            
            with importlib.resources.as_file(json_path_obj) as json_path:
                success = self._cpp_tree.load_data(str(json_path))
                if not success:
                    raise RuntimeError(f"Failed to load VAD data from {json_path}")
                    
        except FileNotFoundError:
            raise FileNotFoundError("VAD.json not found within the package.")
            
    def search(self, V: float, A: float, D: float, k: int = 5, d: float = 1.0, 
                 SIGMA: float = 0.5, opt: str = "knn") -> dict:
        """
        Searches for nearest emotions in the VAD space.
        
        Args:
            V (float): Valence
            A (float): Arousal
            D (float): Dominance
            k (int): Number of neighbors to find.
            d (float): Radius for search (if using 'knn_d').
            SIGMA (float): Sigma for Gaussian similarity.
            opt (str): Search options (e.g., 'knn', 'knn_d', 'cos', 'gauss_w').

        Returns:
            dict: A dictionary containing the search results.
        """
        json_string_result = self._cpp_tree.VAD_search_near_k(V, A, D, k, d, SIGMA, opt)
        
        return json.loads(json_string_result)

__all__ = ["EGOSearcher"]