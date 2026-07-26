from __future__ import annotations

from typing import Any, Dict, List


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for ranked_list in ranked_lists:
            for rank, document in enumerate(ranked_list, start=1):
                chunk_id = document["chunk_id"]
                if chunk_id not in fused:
                    fused[chunk_id] = document.copy()
                    fused[chunk_id]["rrf_score"] = 0.0
                for score_name in ("dense_score", "sparse_score", "graph_score"):
                    if document.get(score_name) is not None:
                        fused[chunk_id][score_name] = document[score_name]
                fused[chunk_id]["rrf_score"] += 1.0 / (self.k + rank)
        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)

