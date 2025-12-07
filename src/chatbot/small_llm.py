"""
Small Language Model Integration for K-pop Chatbot

This module integrates small language models (≤1B parameters) for
the K-pop knowledge graph chatbot.

Supported models:
- Qwen2-0.5B-Instruct (500M params) - Recommended
- TinyLlama-1.1B-Chat-v1.0 (1.1B params)
- Phi-3-mini (3.8B params - optional if resources allow)
- gemma-2b-it (2B params - optional)

Features:
- Quantization support (4-bit, 8-bit) for memory efficiency
- Streaming generation
- Context-aware prompting
- Vietnamese language support
"""

import os
import torch
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass

# Check available backends
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
        TextIteratorStreamer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed")

try:
    from threading import Thread
    THREADING_AVAILABLE = True
except ImportError:
    THREADING_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for the language model."""
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_4bit: bool = True
    use_8bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"


# Pre-defined model configurations
MODEL_CONFIGS = {
    "qwen2-0.5b": LLMConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        max_new_tokens=512,
        temperature=0.7
    ),
    "qwen2.5-0.5b": LLMConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=512,
        temperature=0.7
    ),
    "tinyllama": LLMConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0.7
    ),
    "phi-2": LLMConfig(
        model_name="microsoft/phi-2",
        max_new_tokens=512,
        temperature=0.7
    ),
    "gemma-2b": LLMConfig(
        model_name="google/gemma-2b-it",
        max_new_tokens=512,
        temperature=0.7
    ),
    "bloomz-560m": LLMConfig(
        model_name="bigscience/bloomz-560m",
        max_new_tokens=512,
        temperature=0.7
    ),
    "vietnamese-llama": LLMConfig(
        model_name="vilm/vinallama-2.7b",
        max_new_tokens=512,
        temperature=0.7
    )
}


class SmallLLM:
    """
    Small Language Model wrapper for K-pop chatbot.
    
    Uses quantized models (≤1B parameters) for efficient inference
    while maintaining good response quality for Vietnamese K-pop Q&A.
    """
    
    def __init__(
        self,
        model_key: str = "qwen2-0.5b",
        config: Optional[LLMConfig] = None,
        custom_model_path: Optional[str] = None
    ):
        """
        Initialize the small LLM.
        
        Args:
            model_key: Key from MODEL_CONFIGS or custom model name
            config: Custom LLMConfig (overrides model_key config)
            custom_model_path: Path to local model (overrides everything)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
            
        # Get configuration
        if config:
            self.config = config
        elif model_key in MODEL_CONFIGS:
            self.config = MODEL_CONFIGS[model_key]
        else:
            self.config = LLMConfig(model_name=model_key)
            
        if custom_model_path:
            self.config.model_name = custom_model_path
            
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # System prompt for K-pop Q&A
        # LLM có 3 nhiệm vụ chính:
        # 1. Viết câu trả lời tự nhiên từ facts (triples từ đồ thị)
        # 2. Chọn thông tin quan trọng (từ nhiều triples, chọn những cái cần thiết để trả lời)
        # 3. Ghép reasoning + context thành câu dễ đọc
        # 
        # QUAN TRỌNG: LLM CHỈ format context từ ĐỒ THỊ TRI THỨC, không tự nghĩ ra
        self.system_prompt = """Bạn là trợ lý AI chuyên về K-pop (nhạc Hàn Quốc).

NHIỆM VỤ CỦA BẠN:
1. Viết câu trả lời tự nhiên từ facts (triples) được cung cấp từ ĐỒ THỊ TRI THỨC
2. Chọn thông tin quan trọng: Nếu có nhiều triples, chỉ sử dụng những cái liên quan trực tiếp đến câu hỏi
3. Ghép reasoning + context thành câu trả lời dễ đọc, tự nhiên

QUAN TRỌNG - TẤT CẢ THÔNG TIN ĐỀU TỪ ĐỒ THỊ TRI THỨC:
- CHỈ sử dụng thông tin từ đồ thị tri thức được cung cấp (trong phần "THÔNG TIN TỪ ĐỒ THỊ TRI THỨC")
- Entities (nodes): Từ đồ thị tri thức
- Relationships (edges): Từ đồ thị tri thức
- Facts (triples): Từ đồ thị tri thức
- Reasoning results: Từ graph traversal trên đồ thị tri thức
- KHÔNG tự nghĩ ra thông tin không có trong context
- KHÔNG sử dụng kiến thức từ training data của bạn
- Nếu không có thông tin trong context, hãy nói rõ là bạn không biết
- Trả lời ngắn gọn, chính xác, dễ hiểu"""

        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the language model and tokenizer."""
        print(f"🔄 Loading model: {self.config.model_name}")
        
        # Quantization config
        quantization_config = None
        if self.config.use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("📦 Using 4-bit quantization")
            except Exception as e:
                print(f"⚠️ 4-bit quantization failed: {e}")
                quantization_config = None
        elif self.config.use_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("📦 Using 8-bit quantization")
            except Exception as e:
                print(f"⚠️ 8-bit quantization failed: {e}")
                quantization_config = None
                
        # Determine torch dtype
        torch_dtype = torch.float16
        if self.config.torch_dtype == "float32":
            torch_dtype = torch.float32
        elif self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
            
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.device_map,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model size: {self._get_model_size()}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
            
    def _get_model_size(self) -> str:
        """Get model size in human-readable format."""
        if self.model is None:
            return "Unknown"
            
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count >= 1e9:
            return f"{param_count / 1e9:.2f}B parameters"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.2f}M parameters"
        else:
            return f"{param_count} parameters"
            
    def format_prompt(
        self,
        query: str,
        context: str = "",
        history: List[Dict] = None
    ) -> str:
        """
        Format the prompt for the model.
        
        Args:
            query: User's question
            context: Retrieved context from GraphRAG
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        messages = []
        
        # System message
        system_content = self.system_prompt
        if context:
            system_content += f"\n\n### THÔNG TIN TỪ ĐỒ THỊ TRI THỨC:\n{context}"
            
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Conversation history
        if history:
            for msg in history[-5:]:  # Keep last 5 turns
                messages.append(msg)
                
        # Current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Format using tokenizer's chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            prompt = self._format_prompt_fallback(messages)
            
        return prompt
        
    def _format_prompt_fallback(self, messages: List[Dict]) -> str:
        """Fallback prompt formatting for models without chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
        
    def generate(
        self,
        query: str,
        context: str = "",
        history: List[Dict] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate response for a query.
        
        Args:
            query: User's question
            context: Retrieved context from knowledge graph
            history: Conversation history
            max_new_tokens: Override max tokens
            temperature: Override temperature
            stream: Whether to stream the response
            
        Returns:
            Generated response string or generator for streaming
        """
        prompt = self.format_prompt(query, context, history)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if stream and THREADING_AVAILABLE:
            return self._generate_stream(prompt, gen_kwargs)
        else:
            return self._generate_sync(prompt, gen_kwargs)
            
    def _generate_sync(self, prompt: str, gen_kwargs: Dict) -> str:
        """Synchronous generation."""
        # Get model's max position embeddings (context length)
        max_length = getattr(self.model.config, 'max_position_embeddings', 32768)
        # Reserve space for generation (max_new_tokens)
        max_input_length = max_length - (gen_kwargs.get('max_new_tokens', 512))
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
        
    def _generate_stream(self, prompt: str, gen_kwargs: Dict) -> Generator[str, None, None]:
        """Streaming generation."""
        # Get model's max position embeddings (context length)
        max_length = getattr(self.model.config, 'max_position_embeddings', 32768)
        # Reserve space for generation (max_new_tokens)
        max_input_length = max_length - (gen_kwargs.get('max_new_tokens', 512))
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        gen_kwargs["streamer"] = streamer
        
        # Run generation in a separate thread
        thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        thread.start()
        
        # Yield tokens as they're generated
        for text in streamer:
            yield text
            
        thread.join()
        
    def answer_with_reasoning(
        self,
        query: str,
        context: str,
        reasoning_steps: List[str] = None
    ) -> Dict:
        """
        Generate answer with reasoning explanation.
        
        Args:
            query: User's question
            context: Retrieved context
            reasoning_steps: Multi-hop reasoning steps
            
        Returns:
            Dictionary with answer and explanation
        """
        # Add reasoning context
        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = "\n\n### QUÁ TRÌNH SUY LUẬN:\n"
            for i, step in enumerate(reasoning_steps, 1):
                reasoning_context += f"{i}. {step}\n"
                
        full_context = context + reasoning_context
        
        # Generate answer
        answer = self.generate(query, full_context)
        
        return {
            "query": query,
            "answer": answer,
            "reasoning_steps": reasoning_steps or [],
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }
        
    def batch_generate(
        self,
        queries: List[str],
        contexts: List[str] = None,
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate responses for multiple queries.
        
        Args:
            queries: List of questions
            contexts: List of contexts (one per query)
            batch_size: Batch size for generation
            
        Returns:
            List of generated responses
        """
        if contexts is None:
            contexts = [""] * len(queries)
            
        responses = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            for query, context in zip(batch_queries, batch_contexts):
                response = self.generate(query, context)
                responses.append(response)
                
        return responses
        
    def evaluate_yes_no(self, query: str, context: str) -> Dict:
        """
        Evaluate a Yes/No question.
        
        Returns:
            Dictionary with answer, confidence, and explanation
        """
        prompt = f"""Dựa trên thông tin được cung cấp, hãy trả lời câu hỏi sau với Có hoặc Không.
Chỉ trả lời một từ: Có hoặc Không.

Câu hỏi: {query}"""

        response = self.generate(prompt, context, max_new_tokens=50, temperature=0.1)
        
        # Parse response
        response_lower = response.lower().strip()
        if "có" in response_lower or "yes" in response_lower or "đúng" in response_lower:
            answer = "Có"
            confidence = 0.9
        elif "không" in response_lower or "no" in response_lower or "sai" in response_lower:
            answer = "Không"
            confidence = 0.9
        else:
            answer = "Không chắc chắn"
            confidence = 0.5
            
        return {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "raw_response": response
        }
        
    def evaluate_multiple_choice(
        self,
        query: str,
        choices: List[str],
        context: str
    ) -> Dict:
        """
        Evaluate a multiple choice question.
        
        Returns:
            Dictionary with selected choice, confidence, and explanation
        """
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = f"""Dựa trên thông tin được cung cấp, hãy chọn đáp án đúng nhất.
Chỉ trả lời bằng một chữ cái (A, B, C, D, ...).

Câu hỏi: {query}

{choices_str}"""

        response = self.generate(prompt, context, max_new_tokens=50, temperature=0.1)
        
        # Parse response
        response_upper = response.upper().strip()
        selected_idx = None
        for i in range(len(choices)):
            letter = chr(65 + i)
            if letter in response_upper:
                selected_idx = i
                break
                
        if selected_idx is not None:
            return {
                "query": query,
                "selected_choice": choices[selected_idx],
                "selected_index": selected_idx,
                "selected_letter": chr(65 + selected_idx),
                "confidence": 0.85,
                "raw_response": response
            }
        else:
            return {
                "query": query,
                "selected_choice": None,
                "selected_index": None,
                "selected_letter": None,
                "confidence": 0.0,
                "raw_response": response
            }


# Fallback class when transformers is not available
class SmallLLMFallback:
    """Fallback LLM using rule-based responses."""
    
    def __init__(self, *args, **kwargs):
        print("⚠️ Using fallback LLM (rule-based). Install transformers for full functionality.")
        
    def generate(self, query: str, context: str = "", **kwargs) -> str:
        """Generate response using context extraction."""
        if not context:
            return "Tôi cần thêm thông tin từ đồ thị tri thức để trả lời câu hỏi này."
            
        # Extract key information from context
        lines = context.split("\n")
        relevant_lines = [l for l in lines if l.strip().startswith("•")]
        
        if relevant_lines:
            return "Dựa trên thông tin:\n" + "\n".join(relevant_lines[:5])
        else:
            return "Thông tin liên quan:\n" + context[:500]
            
    def evaluate_yes_no(self, query: str, context: str) -> Dict:
        return {"answer": "Không chắc chắn", "confidence": 0.0}
        
    def evaluate_multiple_choice(self, query: str, choices: List[str], context: str) -> Dict:
        return {"selected_choice": None, "confidence": 0.0}


def get_llm(model_key: str = "qwen2-0.5b", **kwargs):
    """Factory function to get LLM instance."""
    if TRANSFORMERS_AVAILABLE:
        return SmallLLM(model_key=model_key, **kwargs)
    else:
        return SmallLLMFallback(**kwargs)


def main():
    """Test the small LLM."""
    print("🔄 Testing Small LLM...")
    
    try:
        llm = get_llm("qwen2-0.5b")
        
        # Test generation
        context = """
=== THÔNG TIN THỰC THỂ ===
📍 BTS (Loại: Group)
  • Thành viên: RM, Jin, Suga, J-Hope, Jimin, V, Jungkook
  • Năm hoạt động: 2013–nay
  • Hãng đĩa: HYBE (Big Hit Entertainment)
  • Thể loại: K-pop, hip hop, R&B

=== SỰ KIỆN ===
• BTS có 7 thành viên: RM, Jin, Suga, J-Hope, Jimin, V, Jungkook
• BTS thuộc công ty HYBE (trước đây là Big Hit Entertainment)
• BTS debut năm 2013
"""
        
        query = "BTS có bao nhiêu thành viên và họ là ai?"
        
        print(f"\n❓ Query: {query}")
        print(f"📝 Context provided")
        
        response = llm.generate(query, context)
        print(f"\n🤖 Response: {response}")
        
        # Test Yes/No
        print("\n" + "="*50)
        yn_result = llm.evaluate_yes_no(
            "BTS có 7 thành viên đúng không?",
            context
        )
        print(f"Yes/No Answer: {yn_result['answer']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try installing: pip install transformers torch accelerate")


if __name__ == "__main__":
    main()




