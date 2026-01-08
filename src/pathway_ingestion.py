"""
Pathway-based data ingestion and vector store for narrative texts.
"""
import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer


class NarrativeChunker:
    """Smart semantic chunker for long narratives."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 500):
        """Initialize chunker."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, strategy: str = "semantic") -> List[Dict[str, Any]]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Full narrative text
            strategy: 'semantic', 'fixed', or 'hybrid'
        
        Returns:
            List of chunks with metadata
        """
        if strategy == "semantic":
            return self._semantic_chunk(text)
        elif strategy == "fixed":
            return self._fixed_chunk(text)
        elif strategy == "hybrid":
            return self._hybrid_chunk(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Ensure we don't break mid-sentence
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.min_chunk_size:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text.strip(),
                'start_pos': start,
                'end_pos': end,
                'length': len(chunk_text)
            })
            
            chunk_id += 1
            start = end - self.overlap
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Semantic chunking based on paragraph/scene boundaries."""
        # Split by double newlines (paragraphs) or chapter markers
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and len(current_chunk) > self.min_chunk_size:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(current_chunk),
                    'length': len(current_chunk)
                })
                chunk_id += 1
                
                # Add overlap from previous chunk
                words = current_chunk.split()
                overlap_words = words[-int(self.overlap / 5):]  # Approximate word overlap
                current_chunk = ' '.join(overlap_words) + ' ' + para
                start_pos = start_pos + len(current_chunk) - len(' '.join(overlap_words)) - len(para)
            else:
                current_chunk += ' ' + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'start_pos': start_pos,
                'end_pos': start_pos + len(current_chunk),
                'length': len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _hybrid_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Hybrid approach: semantic first, then fixed if chunks too large."""
        semantic_chunks = self._semantic_chunk(text)
        
        # Further split any chunks that are too large
        final_chunks = []
        chunk_id = 0
        
        for chunk in semantic_chunks:
            if chunk['length'] <= self.chunk_size * 1.5:
                chunk['chunk_id'] = chunk_id
                final_chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large chunk with fixed strategy
                sub_chunks = self._fixed_chunk(chunk['text'])
                for sub_chunk in sub_chunks:
                    sub_chunk['chunk_id'] = chunk_id
                    sub_chunk['start_pos'] += chunk['start_pos']
                    sub_chunk['end_pos'] += chunk['start_pos']
                    final_chunks.append(sub_chunk)
                    chunk_id += 1
        
        logger.info(f"Created {len(final_chunks)} hybrid chunks")
        return final_chunks


class PathwayVectorStore:
    """Pathway-based vector store for narrative retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Pathway vector store."""
        self.config = config
        self.embedding_model_name = config.get('embeddings', {}).get('model', 'BAAI/bge-large-en-v1.5')
        self.dimension = config.get('pathway', {}).get('vector_store', {}).get('dimension', 1024)
        self.collection_name = config.get('pathway', {}).get('vector_store', {}).get('collection_name', 'narrative_chunks')
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.chunker = NarrativeChunker(
            chunk_size=config.get('pathway', {}).get('chunking', {}).get('chunk_size', 1000),
            overlap=config.get('pathway', {}).get('chunking', {}).get('chunk_overlap', 200),
            min_chunk_size=config.get('pathway', {}).get('chunking', {}).get('min_chunk_size', 500)
        )
        
        logger.info(f"Initialized PathwayVectorStore with model: {self.embedding_model_name}")
    
    def ingest_narrative(self, narrative_text: str, narrative_id: str, strategy: str = "semantic") -> List[Dict[str, Any]]:
        """
        Ingest a narrative into the vector store.
        
        Args:
            narrative_text: Full text of the narrative
            narrative_id: Unique identifier for the narrative
            strategy: Chunking strategy
        
        Returns:
            List of chunks with embeddings
        """
        logger.info(f"Ingesting narrative {narrative_id} with {len(narrative_text)} characters")
        
        # Chunk the text
        chunks = self.chunker.chunk_text(narrative_text, strategy=strategy)
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Add embeddings and narrative_id to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            chunk['narrative_id'] = narrative_id
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def retrieve(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Query text
            chunks: List of chunks with embeddings
            top_k: Number of top results to return
        
        Returns:
            List of top-k most relevant chunks
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarity scores
        scores = []
        for chunk in chunks:
            score = np.dot(query_embedding, chunk['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding'])
            )
            scores.append(score)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return top-k chunks with scores
        results = []
        for idx in top_indices:
            result = chunks[idx].copy()
            result['score'] = float(scores[idx])
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} chunks for query (scores: {[f'{r['score']:.3f}' for r in results[:3]]})")
        return results
    
    def hybrid_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 20, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Query text
            chunks: List of chunks with embeddings
            top_k: Number of top results to return
            keyword_weight: Weight for keyword matching (0-1)
        
        Returns:
            List of top-k most relevant chunks
        """
        # Semantic search
        semantic_results = self.retrieve(query, chunks, top_k=top_k * 2)
        
        # Keyword matching
        query_terms = set(query.lower().split())
        for result in semantic_results:
            text_terms = set(result['text'].lower().split())
            keyword_score = len(query_terms & text_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores
            result['hybrid_score'] = (1 - keyword_weight) * result['score'] + keyword_weight * keyword_score
        
        # Re-sort by hybrid score
        semantic_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return semantic_results[:top_k]


class Reranker:
    """Cross-encoder reranker for improved retrieval."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker."""
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"Initialized Reranker with model: {model_name}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query text
            documents: List of document chunks
            top_k: Number of top results to return (None = all)
        
        Returns:
            Reranked list of documents
        """
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add rerank scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k:
            documents = documents[:top_k]
        
        logger.info(f"Reranked {len(documents)} documents (top score: {documents[0]['rerank_score']:.3f})")
        return documents
