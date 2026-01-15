
import json
import os
from typing import Dict, Any, List

class MemoryStorage:
    def __init__(self, filepath: str = "memory.json"):
        self.filepath = filepath
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading memory from {self.filepath}: {e}")
                self.data = {}
        else:
            self.data = {}

    def save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memory to {self.filepath}: {e}")

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()

    def delete(self, key: str):
        if key in self.data:
            del self.data[key]
            self.save()

    def search(self, query: str) -> List[tuple]:
        """
        Simple keyword search. Returns list of (key, value) tuples.
        """
        query = query.lower()
        results = []
        for k, v in self.data.items():
            # Convert value to string for searching
            str_val = str(v).lower()
            if query in k.lower() or query in str_val:
                results.append((k, v))
        return results

    def list_all(self) -> Dict[str, Any]:
        return self.data
