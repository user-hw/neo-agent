"""neo-agent-kit RAG (检索增强生成) 系统模块

提供文档处理和智能检索能力:
- DocumentProcessor: 多格式文档加载和智能分块
- RAGPipeline: 端到端的索引-检索-生成管道
"""
from .document import Document, DocumentProcessor
from .pipeline import RAGPipeline, create_rag_pipeline

__all__ = [
    "Document",
    "DocumentProcessor",
    "RAGPipeline",
    "create_rag_pipeline",
]
