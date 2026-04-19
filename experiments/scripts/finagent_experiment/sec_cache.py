import hashlib
import json
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class SECQueryCache:
    """SEC EDGAR API query cache, based on file system storage"""

    def __init__(self, cache_dir: str = None):
        """
            cache_dir: cache directory path. default to ~/.cache/finagent_sec_cache/
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "finagent_sec_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._hit_count = 0
        self._miss_count = 0

    def _make_cache_key(self, query: str, form_types, ciks, start_date: str, end_date: str, page, top_n_results) -> str:
        """generate unique cache key (SHA256 hash) based on query parameters

        normalize parameters for consistency (sort lists, remove duplicates), ensure same semantics of query get same key.
        """
        # normalize parameters for consistency (sort lists, remove duplicates), ensure same semantics of query get same key.
        normalized = {
            "query": str(query).strip().lower(),
            "form_types": sorted([str(ft).strip().upper() for ft in (form_types or [])]),
            "ciks": sorted([str(c).strip() for c in (ciks or [])]),
            "start_date": str(start_date).strip(),
            "end_date": str(end_date).strip(),
            "page": str(page).strip(),
            "top_n_results": int(top_n_results) if top_n_results else 0,
        }
        key_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """return cache file path. use two-level directory structure to avoid too many files in a single directory"""
        # use first 2 characters of hash as subdirectory
        subdir = cache_key[:2]
        cache_path = os.path.join(self.cache_dir, subdir, f"{cache_key}.json")
        return cache_path

    def get(self, query, form_types, ciks, start_date, end_date, page, top_n_results) -> Optional[Any]:
        """search for cache. return cached data on hit, return None on miss"""
        cache_key = self._make_cache_key(query, form_types, ciks, start_date, end_date, page, top_n_results)
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self._hit_count += 1
                logger.info(f"[SEC Cache] HIT (total hits: {self._hit_count}, misses: {self._miss_count}) key={cache_key[:12]}...")
                return cached_data.get("result")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[SEC Cache] Failed to read cache file {cache_path}: {e}")
                return None

        self._miss_count += 1
        logger.info(f"[SEC Cache] MISS (total hits: {self._hit_count}, misses: {self._miss_count}) key={cache_key[:12]}...")
        return None

    def put(self, query, form_types, ciks, start_date, end_date, page, top_n_results, result) -> None:
        """store successful query result in cache"""
        cache_key = self._make_cache_key(query, form_types, ciks, start_date, end_date, page, top_n_results)
        cache_path = self._get_cache_path(cache_key)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        cache_data = {
            "params": {
                "query": query,
                "form_types": form_types,
                "ciks": ciks,
                "start_date": start_date,
                "end_date": end_date,
                "page": page,
                "top_n_results": top_n_results,
            },
            "result": result,
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[SEC Cache] STORED key={cache_key[:12]}...")
        except IOError as e:
            logger.warning(f"[SEC Cache] Failed to write cache file {cache_path}: {e}")

    @property
    def stats(self) -> dict:
        """return cache statistics"""
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / max(1, self._hit_count + self._miss_count),
        }
