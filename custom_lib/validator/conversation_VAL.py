import json
from json_repair import repair_json
from pydantic import BaseModel, ValidationError
from typing import Union
from output import UserOut_error

# get result as str
def is_this_json(raw_data_llm:str) -> Union[dict, UserOut_error]:
    try:
        # is this json?
        json.loads(raw_data_llm)
        print("--- level 1: json format is correct! ---")
    except:
        try:
            # if json is contaminated, try to fix it
            repaired_dict:dict = repair_json(raw_data_llm, return_objects=True)
            print("--- level 1: json repair complete! --- ")
        except Exception as e_repair:
            # it failed :(
            print(f"--- level 1: json repair failed! --- \n{e_repair}")
            return UserOut_error(
                error_location="VAL_llm_json,  level 1",
                error_detail=f"Failed to repair json. Json file was contaminated too much. {e_repair}"
                )
    # validate every type of json