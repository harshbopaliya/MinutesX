"""
MemoryBank - Long-term memory storage for meeting data.

Provides:
- Vector-based semantic search
- Document storage and retrieval
- Memory compaction for old meetings
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from config import config
from observability.logger import get_logger


logger = get_logger(__name__)


class MemoryBank:
    """
    Long-term memory bank for storing and retrieving meeting data.
    
    Supports multiple backends:
    - local: Simple file-based storage with in-memory search
    - chromadb: ChromaDB vector store
    - faiss: FAISS vector index
    """
    
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or config.memory.backend
        self.storage_path = Path(config.memory.chromadb_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._vector_store = None
        
        self._initialize_backend()
        self._load_documents()
        
        logger.info(f"MemoryBank initialized with backend: {self.backend}")
    
    def _initialize_backend(self):
        """Initialize the vector store backend."""
        if self.backend == "chromadb":
            try:
                import chromadb
                
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.storage_path / "chromadb")
                )
                self._vector_store = self._chroma_client.get_or_create_collection(
                    name="meetings",
                    metadata={"description": "MinutesX meeting memory"}
                )
                logger.info("ChromaDB backend initialized")
            except ImportError:
                logger.warning("ChromaDB not installed, falling back to local")
                self.backend = "local"
            except Exception as e:
                logger.error(f"ChromaDB initialization failed: {e}")
                self.backend = "local"
        
        elif self.backend == "faiss":
            try:
                import faiss
                import numpy as np
                
                # Initialize FAISS index
                self._faiss_dimension = 768  # Default embedding dimension
                self._faiss_index = faiss.IndexFlatL2(self._faiss_dimension)
                self._faiss_id_map: List[str] = []
                logger.info("FAISS backend initialized")
            except ImportError:
                logger.warning("FAISS not installed, falling back to local")
                self.backend = "local"
    
    def _load_documents(self):
        """Load existing documents from storage."""
        docs_file = self.storage_path / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file, 'r') as f:
                    self._documents = json.load(f)
                logger.info(f"Loaded {len(self._documents)} documents from storage")
            except Exception as e:
                logger.error(f"Failed to load documents: {e}")
    
    def _save_documents(self):
        """Save documents to storage."""
        docs_file = self.storage_path / "documents.json"
        try:
            with open(docs_file, 'w') as f:
                json.dump(self._documents, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def store(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a document in memory.
        
        Args:
            doc_id: Unique document identifier
            text: Searchable text content
            metadata: Additional metadata to store
            
        Returns:
            Success status
        """
        logger.debug(f"Storing document: {doc_id}")
        
        try:
            document = {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "stored_at": datetime.utcnow().isoformat(),
            }
            
            self._documents[doc_id] = document
            
            # Store in vector backend
            if self.backend == "chromadb" and self._vector_store:
                self._vector_store.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata or {}],
                )
            
            self._save_documents()
            logger.info(f"Document stored: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents with scores
        """
        logger.debug(f"Searching: {query[:50]}...")
        
        try:
            if self.backend == "chromadb" and self._vector_store:
                results = self._vector_store.query(
                    query_texts=[query],
                    n_results=limit,
                )
                
                documents = []
                if results and results['ids']:
                    for i, doc_id in enumerate(results['ids'][0]):
                        if doc_id in self._documents:
                            doc = self._documents[doc_id].copy()
                            doc['score'] = 1 - results['distances'][0][i] if results.get('distances') else 1.0
                            documents.append(doc)
                
                return documents
            
            else:
                # Simple keyword search for local backend
                return self._simple_search(query, limit)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _simple_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search for local backend."""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc_id, doc in self._documents.items():
            text = doc.get("text", "").lower()
            # Simple scoring based on word overlap
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored_docs.append((score, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scored_docs[:limit]:
            result = doc.copy()
            result['score'] = score
            results.append(result)
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        return self._documents.get(doc_id)
    
    def update_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing document."""
        if doc_id not in self._documents:
            return False
        
        if text is not None:
            self._documents[doc_id]["text"] = text
        
        if metadata is not None:
            self._documents[doc_id]["metadata"].update(metadata)
        
        self._documents[doc_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save_documents()
        
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._save_documents()
            
            if self.backend == "chromadb" and self._vector_store:
                self._vector_store.delete(ids=[doc_id])
            
            return True
        return False
    
    def get_old_documents(self, days: int) -> List[Dict[str, Any]]:
        """Get documents older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        old_docs = []
        
        for doc in self._documents.values():
            stored_at = doc.get("stored_at", "")
            if stored_at:
                try:
                    doc_date = datetime.fromisoformat(stored_at)
                    if doc_date < cutoff:
                        old_docs.append(doc)
                except ValueError:
                    pass
        
        return old_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            "backend": self.backend,
            "document_count": len(self._documents),
            "storage_path": str(self.storage_path),
        }
