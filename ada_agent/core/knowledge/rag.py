import os
import re

class SimpleRAG:
    def __init__(self, knowledge_dir):
        self.knowledge_dir = knowledge_dir
        self.chunks = []
        self._load_knowledge()

    def _load_knowledge(self):
        """Loads all .txt files from the directory and chunks them."""
        self.chunks = []
        if not os.path.exists(self.knowledge_dir):
            print(f"Warning: Knowledge directory '{self.knowledge_dir}' does not exist.")
            return

        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Simple chunking by paragraphs (double newline)
                    # We also add the filename as context
                    file_chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
                    
                    for chunk in file_chunks:
                        self.chunks.append({
                            "source": filename,
                            "content": chunk
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"[SimpleRAG] Loaded {len(self.chunks)} chunks from {self.knowledge_dir}")

    def retrieve(self, query, top_k=3):
        """
        Retrieves top_k chunks based on simple keyword overlap.
        """
        if not self.chunks:
            return []

        # Normalize query: lowercase and remove non-alphanumeric
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_chunks = []
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            # Simple scoring: count how many query words are in the chunk
            # Bonus: if exact phrase match (naive) or high density
            
            score = 0
            for word in query_words:
                if word in content_lower:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score desc, then alphabetical (for stability)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k contents
        results = [item[1] for item in scored_chunks[:top_k]]
        return results
