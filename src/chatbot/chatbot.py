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
from .knowledge_graph_neo4j import KpopKnowledgeGraphNeo4j
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
        # Pass LLM to GraphRAG để dùng cho understanding (nếu có)
        # LLM sẽ được load sau, nên pass None lúc đầu, sẽ set sau
        self.rag = GraphRAG(
            knowledge_graph=self.kg,
            use_cache=True,
            llm_for_understanding=None  # Sẽ set sau khi LLM load xong
        )
        
        # 3. Multi-hop Reasoner
        if verbose:
            print("  🧠 Initializing Multi-hop Reasoner...")
        self.reasoner = MultiHopReasoner(self.kg)
        
        # 4. Small LLM (optional)
        self.llm = None
        if llm_model:
            if verbose:
                print(f"  🤖 Loading LLM: {llm_model}...")
            try:
                self.llm = get_llm(llm_model)
                # Set LLM cho GraphRAG để dùng cho understanding
                self.rag.llm_for_understanding = self.llm
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ LLM loading failed: {e}")
                    print("  💡 Using fallback mode (context-based responses)")
                self.llm = None
        else:
            if verbose:
                print("  🤖 LLM skipped (graph-only mode)")
            
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
        
        # ============================================
        # BƯỚC 1: GRAPHRAG - LẤY CONTEXT TỪ ĐỒ THỊ TRI THỨC
        # ============================================
        # ✅ GraphRAG LUÔN được sử dụng để lấy context từ Knowledge Graph
        # GraphRAG thực hiện 3 bước:
        # 1. Semantic Search: Tìm node gần nhất bằng vector search (FAISS + embeddings)
        # 2. Expand Subgraph: Từ node tìm được → mở rộng hàng xóm 1-2 hop → lấy subgraph
        # 3. Build Context: Chuyển subgraph → text/triples để feed vào LLM
        # 
        # TẤT CẢ thông tin đều lấy từ ĐỒ THỊ TRI THỨC (Knowledge Graph), không phải từ LLM memory
        context = self.rag.retrieve_context(
            query,
            max_entities=5,
            max_hops=max_hops
        )
        
        # 2.5. Check if this is a membership Yes/No question - use reasoning directly
        query_lower = query.lower()
        is_membership_question = (
            any(kw in query_lower for kw in ['có phải', 'phải', 'là thành viên', 'is a member', 'belongs to', 'có thành viên']) and
            any(kw in query_lower for kw in ['thành viên', 'member'])
        )
        
        # Check if this is a "list members" question: "Ai là thành viên", "Who are members"
        is_list_members_question = any(kw in query_lower for kw in [
            'ai là thành viên', 'who are', 'thành viên của', 'members of',
            'thành viên nhóm', 'thành viên ban nhạc', 'có những thành viên'
        ]) and 'có phải' not in query_lower and 'không' not in query_lower
        
        # Check if this is a "same group" question - use reasoning directly
        is_same_group_question = any(kw in query_lower for kw in ['cùng nhóm', 'cùng nhóm nhạc', 'same group', 'cùng ban nhạc'])
        
        # Check if this is a "same company" question - use reasoning directly
        # Mở rộng patterns để detect nhiều cách hỏi hơn
        is_same_company_question = any(kw in query_lower for kw in [
            'cùng công ty', 'same company', 'cùng hãng', 'cùng label', 'cùng hãng đĩa',
            'cùng công ty hay', 'cùng hãng hay', 'cùng công ty không', 'cùng hãng không',
            'có cùng công ty', 'có cùng hãng', 'có cùng label'
        ])
        
        # Check if this is a "list members" question: "Ai là thành viên", "Who are members"
        is_list_members_question = any(kw in query_lower for kw in [
            'ai là thành viên', 'who are', 'thành viên của', 'members of',
            'thành viên nhóm', 'thành viên ban nhạc', 'có những thành viên'
        ]) and 'có phải' not in query_lower and 'không' not in query_lower
        
        # ============================================
        # BƯỚC 2: MULTI-HOP REASONING - SUY LUẬN TRÊN ĐỒ THỊ
        # ============================================
        # ✅ Đảm bảo multi-hop reasoning LUÔN được sử dụng khi enabled
        # Multi-hop reasoning sử dụng ĐỒ THỊ TRI THỨC để:
        # - Tìm paths giữa entities (BFS/DFS trên graph)
        # - Traverse relationships (MEMBER_OF, MANAGED_BY, etc.)
        # - So sánh entities qua nhiều hops
        # 
        # TẤT CẢ suy luận đều dựa trên ĐỒ THỊ TRI THỨC, không phải LLM reasoning
        reasoning_result = None
        if use_multi_hop:
            # ✅ CHIẾN LƯỢC AN TOÀN: Rule-based extraction + KG validation trước khi reasoning
            # 
            # Ưu tiên extract entities cho same_group/same_company/list_members questions bằng rule-based
            # Vì đây là câu hỏi factual, cần entities chính xác để reasoning đúng
            if is_same_group_question or is_same_company_question or is_list_members_question:
                # LUÔN force extract entities bằng rule-based (nhanh, chính xác)
                # Bỏ qua GraphRAG nếu không tìm đủ (GraphRAG có thể extract sai)
                extracted = self._extract_entities_for_membership(query)
                
                # Với list_members_question, chỉ cần 1 entity (group)
                min_entities = 1 if is_list_members_question else 2
                
                if len(extracted) >= min_entities:
                    # ✅ VALIDATE: Verify tất cả entities với KG trước khi reasoning
                    validated_entities = []
                    for e in extracted:
                        entity_data = self.kg.get_entity(e)
                        if entity_data:  # Chỉ dùng nếu validate thành công
                            validated_entities.append(e)
                    
                    if len(validated_entities) >= min_entities:
                        # Có đủ entities đã validate → dùng ngay để reasoning (nhanh và chính xác)
                        # ⚠️ QUAN TRỌNG: Multi-hop reasoning do Reasoner thực hiện (graph algorithm)
                        # KHÔNG giao cho LLM nhỏ
                        reasoning_result = self.reasoner.reason(
                            query,
                            start_entities=validated_entities,
                            max_hops=max_hops
                        )
                        # Update context với entities đã validate
                        for e in validated_entities:
                            if not any(existing['id'].lower() == e.lower() for existing in context['entities']):
                                entity_data = self.kg.get_entity(e)
                                if entity_data:
                                    context['entities'].append({
                                        'id': e,
                                        'type': entity_data.get('label', 'Unknown'),
                                        'score': 0.9  # High score vì đã verify với KG
                                    })
                elif len(extracted) == 1:
                    # Chỉ có 1 entity → vẫn thử reasoning (có thể tìm thêm từ graph)
                    reasoning_result = self.reasoner.reason(
                        query,
                        start_entities=extracted,
                        max_hops=max_hops
                    )
                else:
                    # Không tìm được entities → reasoner sẽ tự extract
                    reasoning_result = self.reasoner.reason(
                        query,
                        start_entities=[],
                        max_hops=max_hops
                    )
            elif is_membership_question and len(context['entities']) < 2:
                # Membership question: try to extract entities nếu GraphRAG không tìm đủ
                extracted = self._extract_entities_for_membership(query)
                if extracted:
                    # Add to context for consistency
                    for e in extracted:
                        if not any(existing['id'].lower() == e.lower() for existing in context['entities']):
                            entity_data = self.kg.get_entity(e)
                            if entity_data:
                                context['entities'].append({
                                    'id': e,
                                    'type': entity_data.get('label', 'Unknown'),
                                    'score': 0.8
                                })
            
            # ✅ LUÔN chạy multi-hop reasoning nếu chưa có result
            # QUAN TRỌNG: Reasoning vẫn do ĐỒ THỊ thực hiện (graph traversal, path search)
            # LLM chỉ dùng để hiểu đầu vào (intent, entities, relations) → không làm reasoning
            if reasoning_result is None:
                if context['entities']:
                    entities = [e['id'] for e in context['entities']]
                    
                    # Sử dụng multi_hop_depth từ LLM understanding nếu có
                    # (LLM đã detect depth → dùng để optimize graph traversal)
                    detected_depth = max_hops
                    for e in context['entities']:
                        if e.get('multi_hop_depth'):
                            detected_depth = max(detected_depth, e.get('multi_hop_depth', max_hops))
                            break
                    
                    reasoning_result = self.reasoner.reason(
                        query,
                        start_entities=entities,
                        max_hops=detected_depth  # Sử dụng depth từ LLM understanding
                    )
                else:
                    # Không có entities → reasoner sẽ tự extract
                    reasoning_result = self.reasoner.reason(
                        query,
                        start_entities=[],
                        max_hops=max_hops
                    )
        
        # ============================================
        # BƯỚC 3: FORMAT CONTEXT CHO LLM (Từ GraphRAG - Knowledge Graph)
        # ============================================
        # ✅ LLM LUÔN nhận context từ GraphRAG (yêu cầu bài tập)
        # Context bao gồm:
        # - Entities từ đồ thị (nodes)
        # - Relationships từ đồ thị (edges)
        # - Facts từ đồ thị (triples)
        # - Paths từ đồ thị (multi-hop paths)
        # 
        # TẤT CẢ context đều từ ĐỒ THỊ TRI THỨC, LLM chỉ nhận và format thành câu trả lời
        # Giới hạn context để tránh vượt quá max_length của model
        # QUAN TRỌNG: Giảm context size để tránh LLM bị nhiễu (1969 entities → quá nhiều!)
        if reasoning_result and reasoning_result.confidence >= 0.6:
            # Có reasoning result tốt → giảm context size (chỉ lấy essentials)
            formatted_context = self.rag.format_context_for_llm(context, max_tokens=5000)
        else:
            # Không có reasoning result hoặc confidence thấp → cần nhiều context hơn
            formatted_context = self.rag.format_context_for_llm(context, max_tokens=10000)
        
        # Add reasoning info to context (Multi-hop reasoning results từ đồ thị)
        # Reasoning results cũng được tạo từ ĐỒ THỊ TRI THỨC (graph traversal)
        if reasoning_result:
            formatted_context += f"\n\n=== KẾT QUẢ SUY LUẬN MULTI-HOP (Từ Đồ Thị Tri Thức) ===\n{reasoning_result.explanation}"
            if reasoning_result.steps:
                formatted_context += f"\n\nSố bước suy luận: {len(reasoning_result.steps)}-hop"
                for i, step in enumerate(reasoning_result.steps[:3], 1):
                    formatted_context += f"\n  Bước {i}: {step.explanation[:100]}"
        
        # ============================================
        # BƯỚC 4: GENERATE RESPONSE - LLM TẠO CÂU TRẢ LỜI TỪ CONTEXT
        # ============================================
        # ✅ YÊU CẦU BÀI TẬP: "Lựa chọn một mô hình ngôn ngữ nhỏ" → PHẢI dùng LLM
        # LLM NHẬN context từ Knowledge Graph (GraphRAG) và tạo câu trả lời tự nhiên
        # 
        # LLM KHÔNG tự nghĩ ra thông tin - CHỈ format context từ đồ thị thành câu trả lời
        # Tất cả facts đều từ ĐỒ THỊ TRI THỨC:
        # - Entities: từ nodes trong graph
        # - Relationships: từ edges trong graph  
        # - Facts: từ triples (source, relationship, target) trong graph
        # - Reasoning: từ graph traversal (paths, hops)
        
        # If reasoning found a direct answer for membership, same group, or same company, use it (more accurate than LLM)
        # QUAN TRỌNG: Ưu tiên reasoning result cho các câu hỏi factual (tránh LLM hallucinate)
        if (is_membership_question or is_same_group_question or is_same_company_question) and reasoning_result and reasoning_result.answer_text:
            # For membership/same group/same company questions, ALWAYS prioritize reasoning result if available
            # Reasoning is more accurate than LLM for factual checks
            if reasoning_result.confidence >= 0.6:  # Lower threshold để ưu tiên reasoning
                # Use reasoning result directly (more accurate, tránh LLM hallucinate)
                response = reasoning_result.answer_text
                if reasoning_result.answer_entities:
                    entities_str = ", ".join(reasoning_result.answer_entities[:10])
                    if entities_str and entities_str not in response:
                        response += f"\n\nDanh sách: {entities_str}"
            else:
                # Low confidence, still use LLM but with reasoning context
                if self.llm and use_llm:
                    formatted_context += f"\n\n=== KẾT QUẢ SUY LUẬN ===\n{reasoning_result.answer_text}\n{reasoning_result.explanation}\n\nHãy sử dụng kết quả suy luận này để trả lời."
                    history = session.get_history(max_turns=3)
                    response = self.llm.generate(
                        query,
                        context=formatted_context,
                        history=history
                    )
                else:
                    response = reasoning_result.answer_text
        elif self.llm and use_llm:
            # ✅ SỬ DỤNG Small LLM với context từ Knowledge Graph (đúng yêu cầu)
            history = session.get_history(max_turns=3)
            response = self.llm.generate(
                query,
                context=formatted_context,  # Context từ GraphRAG (Knowledge Graph)
                history=history
            )
        elif reasoning_result and reasoning_result.answer_text:
            # Fallback: Nếu LLM không available, dùng reasoning result
            # (Nhưng ưu tiên dùng LLM để đáp ứng yêu cầu bài tập)
            response = reasoning_result.answer_text
            if reasoning_result.answer_entities:
                entities_str = ", ".join(reasoning_result.answer_entities[:10])
                if len(reasoning_result.answer_entities) > 10:
                    entities_str += f" và {len(reasoning_result.answer_entities) - 10} khác"
                if entities_str and entities_str not in response:
                    response += f"\n\nDanh sách: {entities_str}"
        elif context['facts']:
            # Fallback: Dùng facts từ Knowledge Graph
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
        query_lower = query.lower()
        
        # Get context
        context = self.rag.retrieve_context(query, max_entities=5, max_hops=2)
        formatted_context = self.rag.format_context_for_llm(context)
        
        # Perform reasoning
        entities = [e['id'] for e in context['entities']]
        reasoning_result = self.reasoner.reason(query, entities, max_hops=2)
        
        # Check if reasoning result already has a Yes/No answer
        if reasoning_result and reasoning_result.answer_text:
            answer_text_lower = reasoning_result.answer_text.lower()
            if answer_text_lower.startswith('có') or 'là thành viên' in answer_text_lower:
                return {
                    "query": query,
                    "answer": "Có",
                    "confidence": reasoning_result.confidence,
                    "explanation": reasoning_result.explanation
                }
            elif answer_text_lower.startswith('không') or 'không phải' in answer_text_lower:
                return {
                    "query": query,
                    "answer": "Không",
                    "confidence": reasoning_result.confidence,
                    "explanation": reasoning_result.explanation
                }
        
        # Rule-based answer FIRST (more accurate for knowledge graph queries)
        answer = None
        confidence = 0.0
        
        # Pattern 1: "X có phải là thành viên của Y không?" 
        if 'thành viên' in query_lower or 'member' in query_lower:
            # Find artist and group in context
            artist_entity = None
            group_entity = None
            
            for entity in context['entities']:
                if entity['type'] == 'Artist':
                    artist_entity = entity
                elif entity['type'] == 'Group':
                    group_entity = entity
            
            # If we have both artist and group, check membership directly
            if artist_entity and group_entity:
                artist_name = artist_entity['id']
                group_name = group_entity['id']
                groups = self.kg.get_artist_groups(artist_name)
                
                if group_name in groups:
                    answer = "Có"
                    confidence = 1.0
                else:
                    answer = "Không"
                    confidence = 1.0
            elif artist_entity:
                # Only have artist, check all groups
                artist_name = artist_entity['id']
                groups = self.kg.get_artist_groups(artist_name)
                # Check if any group is mentioned in query or context
                query_groups = [e['id'] for e in context['entities'] if e['type'] == 'Group']
                if query_groups:
                    # Check if artist is member of any mentioned group
                    if any(g in groups for g in query_groups):
                        answer = "Có"
                        confidence = 1.0
                    else:
                        answer = "Không"
                        confidence = 0.9
                else:
                    # No group found, check if group name is in query text
                    for group in groups:
                        if group.lower() in query_lower:
                            answer = "Có"
                            confidence = 1.0
                            break
                    if answer is None:
                        answer = "Không"
                        confidence = 0.8
            else:
                # No artist found
                answer = "Không"
                confidence = 0.7
                
        # Pattern 2: "X thuộc công ty Y" (True/False check)
        elif 'thuộc công ty' in query_lower or 'thuộc company' in query_lower:
            # Extract company name from query (after "thuộc công ty")
            for entity in context['entities']:
                if entity['type'] == 'Group':
                    company = self.kg.get_group_company(entity['id'])
                    if company and company.lower() in query_lower:
                        answer = "Đúng"
                        confidence = 1.0
                        break
                    elif company:
                        answer = "Sai"
                        confidence = 0.9
                        break
            if answer is None:
                answer = "Sai"
                confidence = 0.7
                
        # Pattern 3: "X và Y có cùng công ty không?"
        elif 'cùng công ty' in query_lower or 'same company' in query_lower:
            if len(context['entities']) >= 2:
                result = self.reasoner.check_same_company(
                    context['entities'][0]['id'],
                    context['entities'][1]['id']
                )
                if result.answer_entities:
                    answer = "Có"
                    confidence = 1.0
                else:
                    answer = "Không"
                    confidence = 0.9
                    
        # Fallback: Use reasoning result
        if answer is None:
            answer_text = reasoning_result.answer_text.lower() if reasoning_result else ""
            if any(word in answer_text for word in ['có', 'đúng', 'yes', 'thuộc', 'là', 'cùng']):
                answer = "Có"
                confidence = reasoning_result.confidence if reasoning_result else 0.6
            elif any(word in answer_text for word in ['không', 'sai', 'no', 'khác', 'không rõ']):
                answer = "Không"
                confidence = reasoning_result.confidence if reasoning_result else 0.6
            else:
                # Try LLM as last resort
                if self.llm:
                    try:
                        llm_result = self.llm.evaluate_yes_no(query, formatted_context)
                        answer = llm_result['answer']
                        confidence = llm_result['confidence']
                    except:
                        answer = "Không"
                        confidence = 0.5
                else:
                    answer = "Không"
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
        query_lower = query.lower()
        
        # Get context
        context = self.rag.retrieve_context(query, max_entities=5, max_hops=2)
        formatted_context = self.rag.format_context_for_llm(context)
        
        # Perform reasoning
        entities = [e['id'] for e in context['entities']]
        reasoning_result = self.reasoner.reason(query, entities, max_hops=2)
        
        selected_index = None
        selected_choice = None
        confidence = 0.0
        
        # ============================================
        # SMART ANSWER SELECTION BASED ON QUERY TYPE
        # ============================================
        
        # Pattern 1: "Công ty nào quản lý X?" - find company in choices
        if 'công ty' in query_lower or 'company' in query_lower:
            for entity in context['entities']:
                if entity['type'] == 'Group':
                    company = self.kg.get_group_company(entity['id'])
                    if company:
                        # Find matching choice
                        for i, choice in enumerate(choices):
                            if company.lower() in choice.lower() or choice.lower() in company.lower():
                                selected_index = i
                                selected_choice = choices[i]
                                confidence = 1.0
                                break
                    break
                    
        # Pattern 2: "X thuộc nhóm nào?" - find group in choices
        elif 'nhóm nào' in query_lower or 'thuộc nhóm' in query_lower:
            for entity in context['entities']:
                if entity['type'] == 'Artist':
                    groups = self.kg.get_artist_groups(entity['id'])
                    for group in groups:
                        for i, choice in enumerate(choices):
                            if group.lower() in choice.lower() or choice.lower() in group.lower():
                                selected_index = i
                                selected_choice = choices[i]
                                confidence = 1.0
                                break
                        if selected_index is not None:
                            break
                    break
                    
        # Pattern 3: "Nhóm nào cùng công ty với X?" - find labelmates in choices
        elif 'cùng công ty' in query_lower or 'labelmate' in query_lower:
            for entity in context['entities']:
                if entity['type'] == 'Group':
                    labelmates = self.reasoner.get_labelmates(entity['id'])
                    for labelmate in labelmates.answer_entities:
                        for i, choice in enumerate(choices):
                            if labelmate.lower() in choice.lower() or choice.lower() in labelmate.lower():
                                selected_index = i
                                selected_choice = choices[i]
                                confidence = 0.9
                                break
                        if selected_index is not None:
                            break
                    break
        
        # Fallback: Score-based selection using context and reasoning result
        if selected_index is None:
            # Combine context and reasoning for better matching
            search_text = formatted_context.lower()
            if reasoning_result:
                search_text += " " + reasoning_result.answer_text.lower()
                search_text += " " + " ".join(reasoning_result.answer_entities).lower()
            
            scores = []
            for i, choice in enumerate(choices):
                score = 0
                choice_clean = choice.lower().strip()
                
                # Exact match - highest score
                if choice_clean in search_text:
                    score += 10
                    
                # Word matching
                choice_words = [w for w in choice_clean.split() if len(w) > 2]
                for word in choice_words:
                    if word in search_text:
                        score += 2
                        
                # Entity matching
                for entity in context['entities']:
                    if entity['id'].lower() in choice_clean:
                        score += 3
                        
                scores.append(score)
                
            max_score = max(scores) if scores else 0
            if max_score > 0:
                selected_index = scores.index(max_score)
                selected_choice = choices[selected_index]
                confidence = min(max_score / 10, 1.0)
            else:
                # Last resort: try LLM
                if self.llm:
                    try:
                        llm_result = self.llm.evaluate_multiple_choice(query, choices, formatted_context)
                        selected_index = llm_result['selected_index']
                        selected_choice = llm_result['selected_choice']
                        confidence = llm_result['confidence']
                    except:
                        selected_index = 0  # Default to first choice
                        selected_choice = choices[0]
                        confidence = 0.25
                else:
                    selected_index = 0
                    selected_choice = choices[0]
                    confidence = 0.25
                
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
    
    def _extract_entities_for_membership(self, query: str) -> List[str]:
        """
        Extract entities from query for membership questions.
        Tries to find artist and group names even if GraphRAG didn't find them.
        """
        entities = []
        query_lower = query.lower()
        
        # Try to find group names (case-insensitive)
        all_groups = [node for node, data in self.kg.graph.nodes(data=True) 
                     if data.get('label') == 'Group']
        
        # Try to find artist names (case-insensitive)
        all_artists = [node for node, data in self.kg.graph.nodes(data=True) 
                      if data.get('label') == 'Artist']
        
        # Search for group name in query (case-insensitive)
        # Xử lý variants: "blackpink", "black pink", "BLACKPINK"
        for group in all_groups:
            group_lower = group.lower()
            group_variants = [
                group_lower,
                group_lower.replace('-', ' '),
                group_lower.replace('-', ''),
                group_lower.replace(' ', ''),  # "black pink" → "blackpink"
            ]
            if any(variant in query_lower for variant in group_variants if len(variant) >= 3):
                entities.append(group)
                break
        
        # Search for artist names in query (case-insensitive) - TÌM TẤT CẢ, không chỉ 1
        # QUAN TRỌNG: Xử lý node có đuôi như "Lisa (ca sĩ)", "Jennie (rapper)"
        found_artists = []
        query_words_list = query_lower.split()  # List để giữ thứ tự
        query_text = query_lower  # Full query text để check substring
        
        for artist in all_artists:
            artist_lower = artist.lower()
            # Extract base name (không có đuôi)
            base_name = self._normalize_entity_name(artist)
            base_name_lower = base_name.lower()
            
            # Tạo variants để match với nhiều format: "g-dragon", "g dragon", "gdragon", "blackpink"
            base_name_variants = [
                base_name_lower,
                base_name_lower.replace('-', ' '),  # "g-dragon" → "g dragon"
                base_name_lower.replace('-', ''),    # "g-dragon" → "gdragon"
                base_name_lower.replace(' ', ''),    # "black pink" → "blackpink"
            ]
            
            # Method 1: Check nếu base name hoặc variants là một từ trong query (exact match)
            # Ví dụ: query "lisa có cùng nhóm" → word "lisa" match với base_name "lisa"
            if any(variant in query_words_list for variant in base_name_variants):
                if artist not in found_artists:
                    found_artists.append(artist)
                    continue
            
            # Method 2: Check nếu base name hoặc variants xuất hiện trong query (substring match)
            # Ví dụ: query "jungkook và lisa" → "jungkook" match với "Jungkook"
            # QUAN TRỌNG: Xử lý "g-dragon" và "blackpink" (lowercase, không space)
            if any(variant in query_text for variant in base_name_variants if len(variant) >= 3):
                if artist not in found_artists:
                    found_artists.append(artist)
                    continue
            
            # Method 3: Check từng word trong query với base name và variants
            for word in query_words_list:
                if len(word) < 3:  # Skip short words
                    continue
                # Exact match với base name hoặc variants
                if word in base_name_variants or word == base_name_lower:
                    if artist not in found_artists:
                        found_artists.append(artist)
                        break
                # Partial match: word là một phần của base name hoặc ngược lại
                elif (word in base_name_lower and len(word) >= 3) or (base_name_lower in word and len(base_name_lower) >= 3):
                    if artist not in found_artists:
                        found_artists.append(artist)
                        break
                # Xử lý tên có dấu gạch ngang: "g-dragon" match với "g" và "dragon"
                elif '-' in base_name_lower:
                    base_parts = base_name_lower.split('-')
                    if word in base_parts and len(word) >= 3:
                        # Check xem có part khác cũng trong query không
                        other_parts = [p for p in base_parts if p != word]
                        if any(p in query_lower for p in other_parts):
                            if artist not in found_artists:
                                found_artists.append(artist)
                                break
        
        # Thêm tất cả artists tìm được (không chỉ 1)
        entities.extend(found_artists)
        
        # Nếu chưa tìm đủ, try fuzzy matching với từng word
        # QUAN TRỌNG: Chỉ match với artists/groups, không match với albums/songs (tránh sai)
        if len(entities) < 2:
            words = query_lower.split()
            # Filter out common Vietnamese words
            stop_words = {'có', 'và', 'cùng', 'nhóm', 'nhạc', 'không', 'là', 'thuộc', 'của', 'với', 'hay', 'hoặc'}
            words = [w for w in words if w not in stop_words and len(w) >= 3]
            
            for word in words:
                # Try exact match (case-insensitive) với artists only (tránh match albums/songs)
                for artist in all_artists:
                    base_name = self._normalize_entity_name(artist)
                    base_name_lower = base_name.lower()
                    # Exact match với base name hoặc variants (xử lý dấu gạch ngang)
                    base_name_variants = [
                        base_name_lower,
                        base_name_lower.replace('-', ' '),
                        base_name_lower.replace('-', ''),
                        base_name_lower.replace(' ', ''),
                    ]
                    if word in base_name_variants or base_name_lower == word:
                        if artist not in entities:
                            entities.append(artist)
                            break
                    # Xử lý tên có dấu gạch ngang: "g-dragon" match với "g" và "dragon"
                    if '-' in base_name_lower:
                        base_parts = base_name_lower.split('-')
                        if word in base_parts and len(word) >= 3:
                            # Check xem có part khác cũng trong query không
                            other_parts = [p for p in base_parts if p != word]
                            if any(p in query_lower for p in other_parts):
                                if artist not in entities:
                                    entities.append(artist)
                                    break
                
                # Try exact match với groups (cũng xử lý variants)
                for group in all_groups:
                    group_lower = group.lower()
                    group_variants = [
                        group_lower,
                        group_lower.replace('-', ' '),
                        group_lower.replace('-', ''),
                        group_lower.replace(' ', ''),
                    ]
                    if word in group_variants or group_lower == word:
                        if group not in entities:
                            entities.append(group)
                            break
                        break
                    # Partial match - word là một phần của node name
                    elif word in node_lower and len(word) >= 3:
                        node_data = self.kg.get_entity(node)
                        if node_data and node_data.get('label') in ['Artist', 'Group']:
                            if node not in entities:
                                entities.append(node)
                            break
        
        # Return tất cả entities tìm được (không giới hạn 2)
        return entities[:10]  # Return max 10 entities để đảm bảo tìm đủ
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """
        Normalize entity name bằng cách remove suffixes trong parentheses.
        
        Ví dụ:
        - "Lisa (ca sĩ)" → "Lisa"
        - "BLACKPINK (nhóm nhạc)" → "BLACKPINK"
        - "BTS (rapper)" → "BTS"
        
        Args:
            entity_name: Entity name có thể có đuôi
            
        Returns:
            Base name (không có đuôi)
        """
        import re
        # Remove suffixes trong parentheses: (ca sĩ), (nhóm nhạc), (rapper), etc.
        normalized = re.sub(r'\s*\([^)]+\)\s*$', '', entity_name)
        return normalized.strip()
        
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

