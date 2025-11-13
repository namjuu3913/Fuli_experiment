from deltaEGO_VDB import EGOSearcher
import deltaEGO_compute
from typing import Union, List, Dict, TypedDict, Optional, TypeAlias
from pathlib import Path
import time, warnings

# c++ stuct import ------------------------------------------------------------------------------------
try:
    CppVADPoint:        TypeAlias = deltaEGO_compute.VADPoint
    CppEGO_axis:        TypeAlias = deltaEGO_compute.EGO_axis
    CppVariable:        TypeAlias = deltaEGO_compute.variable
    CppWeight:          TypeAlias = deltaEGO_compute.weight
    CppComputeIn:       TypeAlias = deltaEGO_compute.compute_in
    CppAnalysisResult: TypeAlias = deltaEGO_compute.AnalysisResult

except AttributeError:  # failed to load cpp lib 
    warnings.warn("="*50, ImportWarning)
    warnings.warn("WARNING: from C++ module 'deltaEGO_compute', cannot find binding class !(ex. VADPoint).", ImportWarning)
    warnings.warn("="*50, ImportWarning)
    raise ImportError("Cannot find C++ binding class from analizer.")
#------------------------------------------------------------------------------------------------------

# for search
class VAD_search(TypedDict):
    V: float            # valance
    A: float            # arousal
    D: float            # dominance
    k: int
    dis: float          # distance
    sigma: Optional[float]
    api: Optional[str]

# for analysis
# VAD.hpp
class VADPoint(TypedDict):       
    v: float            # valance
    a: float            # arousal
    d: float            # dominance
    timestamp: float    # when does this emotion generated
    owner: str          # owner of this emotion

class VAD_ave(TypedDict):
    x: float
    y: float
    z: float
    radius: float

# input --------------------------------
class weight(TypedDict, total=False): # total=False -> using cpp default
    weightA_stress: float
    weightV_stress: float
    weightV_reward: float
    weightA_reward: float
    weight_k: float

class variable(TypedDict, total=False):
    theta_0: float
    dampening_factor: float

class EGO_axis(TypedDict, total=False):
    baseline: VADPoint
    stabilityRadius: float

class _compute_in_mandatory(TypedDict):
    current: VADPoint
    history: List[VADPoint]

class compute_in(_compute_in_mandatory, total=False):
    prev: Optional[VADPoint]
    emotion_base: Optional[EGO_axis]
    variables: Optional[variable]
    weights: Optional[weight]
# input ---------------------------------

# return --------------------------------
class InstantMetrics(TypedDict):
    stress: float
    reward: float
    ratio_total: float
    stress_ratio: float
    reward_ratio: float
    deviation: float

class DynamicMetrics(TypedDict):
    delta: VADPoint
    affective_lability: float

class CumulativeMetrics(TypedDict):
    average_area: VAD_ave
    stress: float
    reward: float
    total: float
    stress_ratio: float
    reward_ratio: float

class AnalysisResult_py(TypedDict):
    instant: InstantMetrics
    dynamics: DynamicMetrics
    cumulative: CumulativeMetrics
# return --------------------------------

class deltaEGO:
    def __init__(self, character_name:str, save_path:Optional[Path] = None):
        self.ego_character:str = character_name
        self.EGO_save_path: Optional[Path] = save_path #TODO
        
        # state
        self.last_emotion: Optional[Dict] = None
        self.last_emotion_VADPoint: Optional[VADPoint] = None
        self.emotion_history: List[Dict] = []
        self.emotion_history_VADPoint : List[VADPoint] = []
        self.analysis_history: List[AnalysisResult_py] = []

        # cpp modules
        self.ego_searcher = EGOSearcher()
        self.compute = deltaEGO_compute.compute

        # VDB search default
        self.DEFAULT_SIGMA: float = 0.5
        self.DEFAULT_API_OPT: str = "knn~l2 -S"

        # base line for emotion analize
        self.default_axis = EGO_axis(
            baseline = VADPoint(v=0.0, a=0.0, d=0.0, timestamp=0.0, owner="base"),
            stabilityRadius = 0.3
        )
        self.default_weights = weight()     # C++ default
        self.default_variables = variable() # C++ default

        # flags
        self.automatic_analize:bool = False
        self.save: bool = False             #TODO


    def VADsearch(self, in_VAD: VAD_search) -> dict:
        # TODO: save and open emotion save data 
        api_opt = in_VAD.get('api') or self.DEFAULT_API_OPT
        sigma = in_VAD.get('sigma') or self.DEFAULT_SIGMA

        ego_result: dict = self.ego_searcher.search(
            V = in_VAD['V'],
            A = in_VAD['A'],
            D = in_VAD['D'],
            k = in_VAD['k'],
            d = in_VAD['dis'],
            SIGMA = sigma,
            opt = api_opt
        )

        current_vad_point = VADPoint(
            v=in_VAD['V'], 
            a=in_VAD['A'], 
            d=in_VAD['D'],
            timestamp=time.time(), 
            owner=self.ego_character
        )

        # update history
        self.emotion_history.append(ego_result)
        self.emotion_history_VADPoint.append(current_vad_point)

        # last state update
        self.last_emotion = ego_result
        self.last_emotion_VADPoint = current_vad_point

        # does it needs to analize automatically?
        if self.automatic_analize:
            self.analize_VAD(return_analysis=False)

        return ego_result

    def _build_cpp_input_bundle(self, input_dict: compute_in):
        """
        Python TypedDict(compute_in) --> C++ CppComputeIn
        """
        try:
            #  opt dict -> struct(CppEGO_axis)
            current_Ea_dict = input_dict.get('emotion_base') or self.default_axis
            baseline_cpp = CppVADPoint(**current_Ea_dict['baseline'])
            ego_axis_cpp = CppEGO_axis(
                baseline=baseline_cpp,
                stabilityRadius=current_Ea_dict['stabilityRadius']
            )
            
            variables_cpp = CppVariable(**(input_dict.get('variables') or self.default_variables))
            weights_cpp = CppWeight(**(input_dict.get('weights') or self.default_weights))

            # VADPoint(dict) -> CppVADPoint
            current_cpp = CppVADPoint(**input_dict['current'])
            
            # prev(dict or None) -> CppVADPoint or None
            prev_point_dict = input_dict.get('prev')
            prev_cpp = CppVADPoint(**prev_point_dict) if prev_point_dict else None
            
            # hitstory -> List[CppVADPoint]
            history_cpp = [CppVADPoint(**p) for p in input_dict['history']]

            # final result (cppComputeIn)
            return CppComputeIn(
                current=current_cpp,
                history=history_cpp, 
                prev=prev_cpp,
                emotion_base=ego_axis_cpp,
                variables=variables_cpp,
                weights=weights_cpp
            )
        except TypeError as e:
            warnings.warn(f"fatal error while building cpp struct Error: {e}", RuntimeWarning)
            raise e #

    def analize_VAD(self, 
                    weights:            Optional[weight] = None, 
                    variables:          Optional[variable] = None, 
                    emotion_base:       Optional[EGO_axis] = None, 
                    return_analysis:    bool = True, 
                    append_emotion:     bool = True
                    ) -> Union[None, CppAnalysisResult]:
        
        if not self.last_emotion_VADPoint:
            warnings.warn("Warning: No VAD data to analyze. Call VADsearch() first.", RuntimeWarning)
            return None
        
        prev_point_dict: Optional[VADPoint] = None
        if len(self.emotion_history_VADPoint) > 1:
            prev_point_dict = self.emotion_history_VADPoint[-2]

        # data to transfer to C++ module (Python TypedDict)
        input_bundle_dict = compute_in(
            current = self.last_emotion_VADPoint,
            history = self.emotion_history_VADPoint,
            prev = prev_point_dict, 
            emotion_base = emotion_base or self.default_axis,
            variables = variables or self.default_variables,
            weights = weights or self.default_weights
        )
        
        # dict -> cpp struct
        input_bundle_cpp = self._build_cpp_input_bundle(input_bundle_dict)
        
        # call cpp module -> C++ object (AnalysisResultObject)
        analyzed_result: CppAnalysisResult = self.compute(input_bundle_cpp)

        # update
        if append_emotion:
            self.analysis_history.append(self.analysis_cpp_to_py(analyzed_result)) 
        

        if return_analysis:
            return analyzed_result
        else:
            return None

    # cpp -> py class translator
    @staticmethod
    def vadpoint_cpp_to_py(p: CppVADPoint) -> VADPoint:
        return {
            "v": p.v,
            "a": p.a,
            "d": p.d,
            "timestamp": p.timestamp,
            "owner": p.owner,
        }
    def analysis_cpp_to_py(self, res: CppAnalysisResult) -> AnalysisResult_py:
        return {
            "instant": 
            {
                "stress":        res.instant.stress,
                "reward":        res.instant.reward,
                "ratio_total":   res.instant.ratio_total,
                "stress_ratio":  res.instant.stress_ratio,
                "reward_ratio":  res.instant.reward_ratio,
                "deviation":     res.instant.deviation,
            },

            "dynamics": 
            {
                "delta":               self.vadpoint_cpp_to_py(res.dynamics.delta),
                "affective_lability":  res.dynamics.affective_lability,
            },

            "cumulative": 
            {
                "average_area": 
                {
                    "x": res.cumulative.average_area.x,
                    "y": res.cumulative.average_area.y,
                    "z": res.cumulative.average_area.z,
                    "radius": res.cumulative.average_area.radius,
                },
                "stress":        res.cumulative.stress,
                "reward":        res.cumulative.reward,
                "total":         res.cumulative.total,
                "stress_ratio":  res.cumulative.stress_ratio,
                "reward_ratio":  res.cumulative.reward_ratio,
            }
        }