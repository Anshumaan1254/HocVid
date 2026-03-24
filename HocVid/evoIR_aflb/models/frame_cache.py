"""
Frame Cache Layer — EvoIR Stage 1

Converted from frame_cache.ipynb (correct as-is, no bugs).
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, Dict


class FrameCache:
    """
    Frame similarity cache for adaptive video processing.
    Uses cosine similarity of 64×64 thumbnails for cache lookup.
    """
    def __init__(self, cache_size: int = 32, similarity_threshold: float = 0.95,
                 thumbnail_size: int = 64):
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.thumbnail_size = thumbnail_size
        self.cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._next_key = 0
        self.stats = {
            'total_queries': 0, 'cache_hits': 0,
            'cache_misses': 0, 'time_saved_ms': 0.0,
        }

    # Aliases for backward compatibility
    @property
    def hits(self):
        return self.stats['cache_hits']

    @property
    def misses(self):
        return self.stats['cache_misses']

    def _make_thumbnail(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 4:
            frame = frame[0]
        thumb = F.interpolate(
            frame.unsqueeze(0),
            size=(self.thumbnail_size, self.thumbnail_size),
            mode='bilinear', align_corners=False)
        thumb_vec = thumb.reshape(-1)
        thumb_vec = F.normalize(thumb_vec, dim=0)
        return thumb_vec

    def query(self, frame: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], float]:
        self.stats['total_queries'] += 1
        thumb = self._make_thumbnail(frame)
        best_sim = -1.0
        best_key = None
        for key, (cached_thumb, _) in self.cache.items():
            sim = torch.dot(thumb, cached_thumb.to(thumb.device)).item()
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self.similarity_threshold and best_key is not None:
            self.stats['cache_hits'] += 1
            self.cache.move_to_end(best_key)
            _, cached_result = self.cache[best_key]
            return True, cached_result, best_sim
        else:
            self.stats['cache_misses'] += 1
            return False, None, best_sim

    def store(self, frame: torch.Tensor, result: torch.Tensor) -> int:
        thumb = self._make_thumbnail(frame)
        key = self._next_key
        self._next_key += 1
        self.cache[key] = (thumb.cpu(), result.cpu())
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return key

    def process_frame(self, frame, process_fn, **kwargs):
        hit, cached_result, sim = self.query(frame)
        if hit:
            return cached_result.to(frame.device), True
        else:
            result = process_fn(frame, **kwargs)
            self.store(frame, result)
            return result, False

    def get_stats(self) -> Dict[str, float]:
        total = self.stats['total_queries']
        hit_rate = self.stats['cache_hits'] / total if total > 0 else 0
        return {**self.stats, 'hit_rate': hit_rate,
                'cache_occupancy': len(self.cache), 'cache_capacity': self.cache_size}

    def clear(self):
        self.cache.clear()
        self.stats = {'total_queries': 0, 'cache_hits': 0,
                      'cache_misses': 0, 'time_saved_ms': 0.0}

    def __repr__(self):
        stats = self.get_stats()
        return (f"FrameCache(size={self.cache_size}, threshold={self.similarity_threshold}, "
                f"occupancy={stats['cache_occupancy']}, hit_rate={stats['hit_rate']:.1%})")
