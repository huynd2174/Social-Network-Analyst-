"""
Main Chatbot Module for K-pop Knowledge Graph

This module integrates all components:
- Knowledge Graph
- GraphRAG
- Multi-hop Reasoning
- Small LLM

Provides a unified interface for the K-pop chatbot.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from .knowledge_graph import KpopKnowledgeGraph
from .graph_rag import GraphRAG
from .multi_hop_reasoning import MultiHopReasoner, ReasoningResult
from .small_llm import SmallLLM, get_llm, TRANSFORMERS_AVAILABLE


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChatSession:
    """A chat session with history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the session."""
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
    def get_history(self, max_turns: int = 5) -> List[Dict]:
        """Get conversation history for context."""
        history = []
        for msg in self.messages[-max_turns * 2:]:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        return history


class KpopChatbot:
    """
    K-pop Knowledge Graph Chatbot.
    
    Combines GraphRAG retrieval with multi-hop reasoning
    and small LLM generation for answering K-pop questions.
    """
    
    def __init__(
        self,
        data_path: str = "data/merged_kpop_data.json",
        llm_model: str = "qwen2-0.5b",
        use_embeddings: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the chatbot.
        
        Args:
            data_path: Path to merged K-pop data
            llm_model: Model key for small LLM
            use_embeddings: Whether to use semantic embeddings
            verbose: Print initialization progress
        """
        self.verbose = verbose
        self.sessions: Dict[str, ChatSession] = {}
        
        # Initialize components
        if verbose:
            print("🔄 Initializing K-pop Chatbot...")
            
        # 1. Knowledge Graph
        if verbose:
            print("  📊 Loading Knowledge Graph...")
        self.kg = KpopKnowledgeGraph(data_path)
        
        # 2. GraphRAG
        if verbose:
            print("  🔍 Initializing GraphRAG...")
        self.rag = GraphRAG(
            knowledge_graph=self.kg,
            use_cache=True
        )
        
        # 3. Multi-hop Reasoner
        if verbose:
            print("  🧠 Initializing Multi-hop Reasoner...")
        self.reasoner = MultiHopReasoner(self.kg)
        
        # 4. Small LLM
        if verbose:
            print(f"  🤖 Loading LLM: {llm_model}...")
        try:
            self.llm = get_llm(llm_model)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ LLM loading failed: {e}")
                print("  💡 Using fallback mode (context-based responses)")
            self.llm = None
            
        if verbose:
            print("✅ Chatbot initialized successfully!")
            
    def create_session(self, session_id: str = None) -> str:
        """Create a new chat session."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.sessions[session_id] = ChatSession(session_id=session_id)
        return session_id
        
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing session."""
        return self.sessions.get(session_id)
        
    def chat(
        self,
        query: str,
        session_id: str = None,
        use_multi_hop: bool = True,
        max_hops: int = 3,
        return_details: bool = False,
        use_llm: bool = True
    ) -> Dict:
        """
        Process a chat query and return response.
        
        Args:
            query: User's question
            session_id: Session ID for conversation history
            use_multi_hop: Enable multi-hop reasoning
            max_hops: Maximum reasoning hops
            return_details: Include detailed reasoning info
            
        Returns:
            Response dictionary with answer and metadata
        """
        # Get or create session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session_id = self.create_session(session_id)
            session = self.sessions[session_id]
            
        # Add user message
        session.add_message("user", query)
        
        # 1. Retrieve context using GraphRAG
        context = self.rag.retrieve_context(
            query,
            max_entities=5,
            max_hops=max_hops
        )
        
        # 2. Perform multi-hop reasoning if enabled
        reasoning_result = None
        if use_multi_hop and context['entities']:
            entities = [e['id'] for e in context['entities']]
            reasoning_result = self.reasoner.reason(
                query,
                start_entities=entities,
                max_hops=max_hops
            )
            
        # 3. Format context for LLM
        formatted_context = self.rag.format_context_for_llm(context)
        
        # Add reasoning info to context
        if reasoning_result:
            formatted_context += f"\n\n=== KẾT QUẢ SUY LUẬN ===\n{reasoning_result.explanation}"
            
        # 4. Generate response
        # Priority: Use reasoning result first (accurate and fast), then LLM only if needed
        # Check if we have a good reasoning result
        has_good_reasoning = (
            reasoning_result and 
            reasoning_result.answer_text and 
            len(reasoning_result.answer_text) > 10 and
            reasoning_result.confidence > 0.5
        )
        
        # Check if query is simple (members, company, etc.) - should use reasoning
        is_simple_query = any(keyword in query.lower() for keyword in [
            'members', 'thành viên', 'member', 'company', 'công ty', 
            'cùng công ty', 'same company', 'labelmate', 'cùng nhóm'
        ])
        
        if has_good_reasoning and (is_simple_query or not use_llm):
            # Use reasoning result - more accurate for structured queries
            response = reasoning_result.answer_text
            if reasoning_result.answer_entities:
                # Add entity list if available
                entities_str = ", ".join(reasoning_result.answer_entities[:10])
                if len(reasoning_result.answer_entities) > 10:
                    entities_str += f" và {len(reasoning_result.answer_entities) - 10} khác"
                if entities_str and entities_str not in response:
                    response += f"\n\nDanh sách: {entities_str}"
        elif self.llm and use_llm and not is_simple_query:
            # Only use LLM for complex queries, not simple factual ones
            history = session.get_history(max_turns=3)
            response = self.llm.generate(
                query,
                context=formatted_context,
                history=history
            )
        else:
            # Fallback: Use reasoning result or context-based response
            if reasoning_result and reasoning_result.answer_text:
                response = reasoning_result.answer_text
            elif context['facts']:
                response = "Dựa trên đồ thị tri thức:\n" + "\n".join(f"• {f}" for f in context['facts'][:5])
            else:
                response = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong đồ thị tri thức."
                
        # Add assistant message
        session.add_message("assistant", response, {
            "entities": [e['id'] for e in context['entities']],
            "reasoning_type": reasoning_result.reasoning_type.value if reasoning_result else None
        })
        
        # Build response
        result = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "entities_found": len(context['entities']),
            "reasoning_hops": len(reasoning_result.steps) if reasoning_result else 0
        }
        
        if return_details:
            result["context"] = context
            result["reasoning"] = {
                "type": reasoning_result.reasoning_type.value if reasoning_result else None,
                "steps": [
                    {
                        "hop": s.hop_number,
                        "operation": s.operation,
                        "explanation": s.explanation
                    }
                    for s in reasoning_result.steps
                ] if reasoning_result else [],
                "confidence": reasoning_result.confidence if reasoning_result else 0
            }
            result["formatted_context"] = formatted_context
            
        return result
        
    def answer_yes_no(
        self,
        query: str,
        return_details: bool = False
    ) -> Dict:
        """
        Answer a Yes/No question.
        
        Args:
            query: Yes/No question
            return_details: Include detailed info
            
        Returns:
            Answer dictionary
        """
        # Get context
        context = self.rag.retrieve_context(query, max_entities=5, max_hops=2)
        formatted_context = self.rag.format_context_for_llm(context)
        
        # Perform reasoning
        entities = [e['id'] for e in context['entities']]
        reasoning_result = self.reasoner.reason(query, entities, max_hops=2)
        
        # Determine answer
        if self.llm:
            result = self.llm.evaluate_yes_no(query, formatted_context)
            answer = result['answer']
            confidence = result['confidence']
        else:
            # Rule-based answer from reasoning
            answer_text = reasoning_result.answer_text.lower()
            if any(word in answer_text for word in ['có', 'đúng', 'yes', 'thuộc', 'là']):
                answer = "Có"
                confidence = reasoning_result.confidence
            elif any(word in answer_text for word in ['không', 'sai', 'no', 'khác']):
                answer = "Không"
                confidence = reasoning_result.confidence
            else:
                answer = "Không chắc chắn"
                confidence = 0.5
                
        result = {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "explanation": reasoning_result.explanation if reasoning_result else ""
        }
        
        if return_details:
            result["context"] = context
            result["reasoning"] = reasoning_result
            
        return result
        
    def answer_multiple_choice(
        self,
        query: str,
        choices: List[str],
        return_details: bool = False
    ) -> Dict:
        """
        Answer a multiple choice question.
        
        Args:
            query: Question
            choices: List of choices
            return_details: Include detailed info
            
        Returns:
            Answer dictionary
        """
        # Get context
        context = self.rag.retrieve_context(query, max_entities=5, max_hops=2)
        formatted_context = self.rag.format_context_for_llm(context)
        
        # Try to find answer in context
        if self.llm:
            result = self.llm.evaluate_multiple_choice(query, choices, formatted_context)
            selected_choice = result['selected_choice']
            selected_index = result['selected_index']
            confidence = result['confidence']
        else:
            # Rule-based selection
            context_text = formatted_context.lower()
            scores = []
            for i, choice in enumerate(choices):
                score = 0
                choice_words = choice.lower().split()
                for word in choice_words:
                    if len(word) > 2 and word in context_text:
                        score += 1
                scores.append(score)
                
            max_score = max(scores)
            if max_score > 0:
                selected_index = scores.index(max_score)
                selected_choice = choices[selected_index]
                confidence = min(max_score / 3, 1.0)
            else:
                selected_index = None
                selected_choice = None
                confidence = 0.0
                
        result = {
            "query": query,
            "choices": choices,
            "selected_choice": selected_choice,
            "selected_index": selected_index,
            "selected_letter": chr(65 + selected_index) if selected_index is not None else None,
            "confidence": confidence
        }
        
        if return_details:
            result["context"] = context
            result["formatted_context"] = formatted_context
            
        return result
        
    # =========== Specialized Query Methods ===========
    
    def get_group_members(self, group_name: str) -> Dict:
        """Get members of a K-pop group."""
        result = self.reasoner.get_group_members(group_name)
        return {
            "group": group_name,
            "members": result.answer_entities,
            "member_count": len(result.answer_entities),
            "answer": result.answer_text
        }
        
    def get_group_company(self, group_name: str) -> Dict:
        """Get company managing a group."""
        result = self.reasoner.get_company_of_group(group_name)
        return {
            "group": group_name,
            "company": result.answer_entities[0] if result.answer_entities else None,
            "answer": result.answer_text
        }
        
    def check_same_company(self, entity1: str, entity2: str) -> Dict:
        """Check if two entities are under the same company."""
        result = self.reasoner.check_same_company(entity1, entity2)
        return {
            "entity1": entity1,
            "entity2": entity2,
            "same_company": len(result.answer_entities) > 0,
            "common_company": result.answer_entities[0] if result.answer_entities else None,
            "answer": result.answer_text
        }
        
    def get_labelmates(self, entity: str) -> Dict:
        """Get all groups/artists under same company."""
        result = self.reasoner.get_labelmates(entity)
        return {
            "entity": entity,
            "labelmates": result.answer_entities,
            "count": len(result.answer_entities),
            "answer": result.answer_text
        }
        
    def find_path(self, source: str, target: str) -> Dict:
        """Find relationship path between two entities."""
        path = self.kg.find_path(source, target)
        
        if path:
            details = self.kg.get_path_details(path)
            path_str = " → ".join([
                f"{d['entity']}({d['type']})" for d in details
            ])
            return {
                "source": source,
                "target": target,
                "path_found": True,
                "path": path,
                "hops": len(path) - 1,
                "description": path_str
            }
        else:
            return {
                "source": source,
                "target": target,
                "path_found": False,
                "path": [],
                "hops": -1,
                "description": f"Không tìm thấy đường đi từ {source} đến {target}"
            }
            
    def get_statistics(self) -> Dict:
        """Get chatbot and knowledge graph statistics."""
        kg_stats = self.kg.get_statistics()
        return {
            "knowledge_graph": kg_stats,
            "active_sessions": len(self.sessions),
            "llm_available": self.llm is not None,
            "embeddings_available": self.rag.embedder is not None
        }


def main():
    """Test the chatbot."""
    print("="*60)
    print("🎤 K-pop Knowledge Graph Chatbot Demo")
    print("="*60)
    
    # Initialize chatbot
    chatbot = KpopChatbot(
        llm_model="qwen2-0.5b",
        verbose=True
    )
    
    # Print statistics
    print("\n📊 Statistics:")
    stats = chatbot.get_statistics()
    print(f"  Nodes: {stats['knowledge_graph']['total_nodes']}")
    print(f"  Edges: {stats['knowledge_graph']['total_edges']}")
    print(f"  LLM: {'✅' if stats['llm_available'] else '❌'}")
    print(f"  Embeddings: {'✅' if stats['embeddings_available'] else '❌'}")
    
    # Test queries
    test_queries = [
        "BTS có bao nhiêu thành viên?",
        "Công ty nào quản lý BLACKPINK?",
        "BTS và SEVENTEEN có cùng công ty không?",
    ]
    
    print("\n" + "="*60)
    print("🧪 Running Test Queries")
    print("="*60)
    
    session_id = chatbot.create_session()
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = chatbot.chat(query, session_id, return_details=True)
        print(f"🤖 Response: {result['response']}")
        print(f"📍 Entities: {result['entities_found']}, Hops: {result['reasoning_hops']}")
        
    # Test specialized queries
    print("\n" + "="*60)
    print("🧪 Specialized Queries")
    print("="*60)
    
    print("\n👥 BTS Members:")
    result = chatbot.get_group_members("BTS")
    print(f"  {result['answer']}")
    
    print("\n🏢 BTS Company:")
    result = chatbot.get_group_company("BTS")
    print(f"  {result['answer']}")
    
    print("\n🔍 Same Company Check (BTS vs SEVENTEEN):")
    result = chatbot.check_same_company("BTS", "SEVENTEEN")
    print(f"  {result['answer']}")


if __name__ == "__main__":
    main()

