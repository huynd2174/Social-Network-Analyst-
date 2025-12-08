"""
K-pop Knowledge Graph Chatbot

Modules:
- knowledge_graph: Build and manage knowledge graph
- graph_rag: GraphRAG implementation for retrieval
- multi_hop_reasoning: Multi-hop reasoning engine
- small_llm: Integration with small language model
- chatbot: Main chatbot interface
- evaluation: Evaluation dataset generator
- comparison: Comparison with other chatbots
"""

from .knowledge_graph import KpopKnowledgeGraph
from .graph_rag import GraphRAG
from .multi_hop_reasoning import MultiHopReasoner
from .small_llm import SmallLLM
from .chatbot import KpopChatbot
from .evaluation import EvaluationDatasetGenerator
from .comparison import ChatbotComparison

__all__ = [
    'KpopKnowledgeGraph',
    'GraphRAG', 
    'MultiHopReasoner',
    'SmallLLM',
    'KpopChatbot',
    'EvaluationDatasetGenerator',
    'ChatbotComparison'
]






