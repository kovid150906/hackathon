"""
Simple vector store implementation without Pathway dependency.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class SimpleVectorStore:
    """Simple in-memory vector store using sentence transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store."""
        self.config = config
        embeddings_config = config.get('embeddings', {})
        self.model_name = embeddings_config.get('model', 'BAAI/bge-large-en-v1.5')
        self.device = embeddings_config.get('device', 'cpu')
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedder = SentenceTransformer(self.model_name, device=self.device)
        
        # Storage
        self.chunks = []
        self.embeddings = None
        
    def add_narrative(self, text: str, metadata: Optional[Dict] = None):
        """Add narrative text to the vector store."""
        # Simple chunking
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                'text': chunk,
                'metadata': metadata or {},
                'chunk_id': i
            })
        
        # Generate embeddings
        if self.chunks:
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def ingest_narrative(self, text: str, narrative_id: str, strategy: str = "semantic") -> List[Dict[str, Any]]:
        """Ingest narrative and return chunks (API compatible with PathwayVectorStore)."""
        self.clear()
        chunks = self._chunk_text(text)
        
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'metadata': {'narrative_id': narrative_id, 'chunk_id': i},
                'chunk_id': i
            }
            self.chunks.append(chunk_dict)
            chunk_dicts.append(chunk_dict)
        
        # Generate embeddings with progress indication
        if self.chunks:
            logger.info(f"Generating embeddings for {len(chunks)} chunks (this may take 15-30 seconds)...")
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.embedder.encode(texts, 
                                                   convert_to_numpy=True,
                                                   show_progress_bar=True,
                                                   batch_size=32)
            logger.info(f"✅ Ingested {len(chunks)} chunks for narrative {narrative_id}")
        
        return chunk_dicts
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking."""
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            start = end - overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        if not self.chunks or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx]['text'],
                'score': float(similarities[idx]),
                'metadata': self.chunks[idx]['metadata']
            })
        
        return results
    
    def clear(self):
        """Clear the vector store."""
        self.chunks = []
        self.embeddings = None


class Reranker:
    """Cross-encoder reranker for improving retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reranker."""
        self.config = config
        reranker_config = config.get('reranker', {})
        self.enabled = reranker_config.get('enabled', True)
        
        if self.enabled:
            model_name = reranker_config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info(f"Loading reranker model: {model_name}")
            try:
                self.model = CrossEncoder(model_name, local_files_only=True)
                self.final_k = reranker_config.get('final_k', 20)
                logger.info(f"Loaded reranker from cache")
            except Exception as e:
                logger.warning(f"Could not load reranker (using cache-only): {e}. Disabling reranker.")
                self.enabled = False
                self.model = None
        else:
            self.model = None
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results."""
        if not self.enabled or not chunks:
            return chunks
        
        # Prepare pairs
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Add scores and sort
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
        
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:self.final_k]
