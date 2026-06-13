# src/fusion.py
from typing import List, Dict, Any, Protocol

class FusionStrategy(Protocol):
    def fuse(self, list_of_ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]: ...

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, list_of_ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        fused_map = {}
        for ranked_list in list_of_ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                cid = doc["chunk_id"]
                if cid not in fused_map:
                    fused_map[cid] = doc.copy()
                    fused_map[cid]["rrf_score"] = 0.0
                
                # Consolidate raw scores for debugging visibility
                for score_type in ["dense_score", "sparse_score", "graph_score"]:
                    if doc.get(score_type) is not None:
                        fused_map[cid][score_type] = doc[score_type]
                        
                fused_map[cid]["rrf_score"] += 1.0 / (self.k + rank)
                
        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_list