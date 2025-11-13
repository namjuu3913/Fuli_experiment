import pprint
from deltaEGO_VDB import EGOSearcher # custom package

print("--- Python Test Script Started ---")

try:
    # VAD class
    searcher = EGOSearcher()
    
    print("VADSearcher loaded successfully.")
    print("VAD.json data file seems to be loaded correctly.")

    # call C++ VAD_search_near_k
    # 'happiness' (V: 0.8, A: 0.6, D: 0.7)
    results = searcher.search(
        V=0.8, 
        A=0.6, 
        D=0.7, 
        k=3, 
        opt="knn~cos" # cosine sim
    )

    print("\n--- Search Results ---")
    pprint.pprint(results)
    print("------------------------")

except Exception as e:
    print(f"\n--- !!! AN ERROR OCCURRED !!! ---")
    print(e)
    print("---------------------------------")