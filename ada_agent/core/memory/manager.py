import os
from .storage import MemoryStorage

class MemoryManager:
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Use user home directory
            home = os.path.expanduser("~")
            ada_dir = os.path.join(home, ".ada")
            if not os.path.exists(ada_dir):
                os.makedirs(ada_dir)
            storage_path = os.path.join(ada_dir, "memory.json")
            
        self.storage = MemoryStorage(storage_path)

    def remember(self, content: str, key: str = None):
        """
        Stores a memory. 
        If key is provided, uses it. 
        If not, we might need to generate one or just use the content itself as a list item (not supported by simple KV storage yet).
        For this MVP, we require a key-value pair concept, or we treat 'key' as the 'topic'.
        """
        if not key:
            # Simple fallback: use timestamp or hash, but let's enforce key for now for 'facts'
            # Or better, user provides "Remember that [My Name] is [Chen]" -> Key: My Name, Value: Chen.
            raise ValueError("Key is required for memory storage in this MVP.")
        
        self.storage.set(key, content)
        return f"Memory stored: [{key}] = {content}"

    def recall(self, query: str) -> str:
        """
        Searches memory for the query.
        """
        results = self.storage.search(query)
        if not results:
            return "No relevant memories found."
        
        # Format results
        output = []
        for k, v in results:
            output.append(f"- {k}: {v}")
        return "\n".join(output)

    def forget(self, key: str):
        val = self.storage.get(key)
        if val:
            self.storage.delete(key)
            return f"Forgot: {key}"
        return f"Nothing found for: {key}"
