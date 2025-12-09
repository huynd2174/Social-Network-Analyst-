"""
Chatbot Comparison Module

This module compares the K-pop Knowledge Graph Chatbot with
popular chatbots on the market (ChatGPT, Gemini, etc.)

Evaluation metrics:
- Accuracy: Correct answers / Total questions
- Precision: For each category
- Response time
- Multi-hop reasoning capability
"""

import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests

# Optional: OpenAI API for ChatGPT comparison
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional: Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    question_type: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    confidence: float
    response_time: float  # seconds
    hops: int
    category: str


@dataclass
class ChatbotEvaluation:
    """Overall evaluation of a chatbot."""
    chatbot_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    accuracy_by_hops: Dict[int, float]
    accuracy_by_type: Dict[str, float]
    accuracy_by_category: Dict[str, float]
    avg_response_time: float
    evaluated_at: str
    results: List[EvaluationResult]


class ChatbotComparison:
    """
    Compare K-pop chatbot with other chatbots.
    
    Supported comparisons:
    - ChatGPT (OpenAI API)
    - Gemini (Google API)
    - Claude (Anthropic API)
    - Local baseline (no knowledge graph)
    """
    
    def __init__(
        self,
        kpop_chatbot=None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None
    ):
        """
        Initialize comparison module.
        
        Args:
            kpop_chatbot: Our K-pop chatbot instance
            openai_api_key: OpenAI API key for ChatGPT
            google_api_key: Google API key for Gemini
        """
        self.kpop_chatbot = kpop_chatbot
        
        # API keys from environment or parameters
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE and self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            # Try to find available model
            try:
                models = genai.list_models()
                available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                
                # Try preferred models (flash models are faster)
                self.gemini_model = None
                preferred = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.0-flash-exp']
                
                for pref in preferred:
                    matching = [m for m in available if pref in m]
                    if matching:
                        try:
                            self.gemini_model = genai.GenerativeModel(matching[0])
                            break
                        except:
                            continue
                
                # If still None, use first available
                if not self.gemini_model and available:
                    model_name = available[0]
                    self.gemini_model = genai.GenerativeModel(model_name)
                elif not self.gemini_model:
                    print(f"⚠️ Warning: No available Gemini models found")
                    self.gemini_model = None
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Gemini model: {e}")
                # Fallback to gemini-pro
                try:
                    self.gemini_model = genai.GenerativeModel('gemini-pro')
                except:
                    self.gemini_model = None
        else:
            self.gemini_model = None
            
    def load_evaluation_dataset(
        self,
        path: str = "data/kpop_eval_2000_multihop_max3hop (1).json"
    ) -> List[Dict]:
        """Load evaluation dataset."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']
        
    def evaluate_kpop_chatbot(
        self,
        questions: List[Dict],
        max_questions: Optional[int] = None
    ) -> ChatbotEvaluation:
        """
        Evaluate our K-pop chatbot.
        
        Args:
            questions: List of evaluation questions
            max_questions: Limit number of questions
            
        Returns:
            ChatbotEvaluation results
        """
        if self.kpop_chatbot is None:
            raise ValueError("K-pop chatbot not initialized")
            
        if max_questions:
            questions = questions[:max_questions]
            
        print(f"🔄 Evaluating K-pop Chatbot on {len(questions)} questions...")
        
        results = []
        correct = 0
        total_time = 0
        
        for i, q in enumerate(questions):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(questions)}")
                
            start_time = time.time()
            
            try:
                if q['question_type'] == 'true_false':
                    result = self.kpop_chatbot.answer_yes_no(q['question'], max_hops_override=3)
                    answer_upper = result['answer'].upper()
                    predicted = "Đúng" if any(word in answer_upper for word in ['ĐÚNG', 'TRUE', 'YES', 'CÓ']) else "Sai"
                elif q['question_type'] == 'yes_no':
                    result = self.kpop_chatbot.answer_yes_no(q['question'])
                    predicted = result['answer']
                else:  # multiple_choice
                    result = self.kpop_chatbot.answer_multiple_choice(
                        q['question'],
                        q['choices']
                    )
                    predicted = result['selected_letter']
                    
                confidence = result.get('confidence', 0.5)
                
            except Exception as e:
                predicted = "Error"
                confidence = 0.0
                
            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time
            
            is_correct = predicted == q['answer']
            if is_correct:
                correct += 1
                
            results.append(EvaluationResult(
                question_id=q['id'],
                question=q['question'],
                question_type=q['question_type'],
                correct_answer=q['answer'],
                predicted_answer=predicted,
                is_correct=is_correct,
                confidence=confidence,
                response_time=response_time,
                hops=q['hops'],
                category=q['category']
            ))
            
        # Calculate metrics
        accuracy = correct / len(questions) if questions else 0
        
        # Accuracy by hops
        accuracy_by_hops = {}
        for hop in [1, 2, 3]:
            hop_results = [r for r in results if r.hops == hop]
            if hop_results:
                accuracy_by_hops[hop] = sum(1 for r in hop_results if r.is_correct) / len(hop_results)
                
        # Accuracy by type
        accuracy_by_type = {}
        for qtype in ['true_false', 'yes_no', 'multiple_choice']:
            type_results = [r for r in results if r.question_type == qtype]
            if type_results:
                accuracy_by_type[qtype] = sum(1 for r in type_results if r.is_correct) / len(type_results)
                
        # Accuracy by category
        accuracy_by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                accuracy_by_category[cat] = sum(1 for r in cat_results if r.is_correct) / len(cat_results)
                
        return ChatbotEvaluation(
            chatbot_name="K-pop Knowledge Graph Chatbot",
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            accuracy_by_hops=accuracy_by_hops,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_category=accuracy_by_category,
            avg_response_time=total_time / len(questions) if questions else 0,
            evaluated_at=datetime.now().isoformat(),
            results=results
        )
        
    def evaluate_chatgpt(
        self,
        questions: List[Dict],
        model: str = "gpt-3.5-turbo",
        max_questions: Optional[int] = None
    ) -> ChatbotEvaluation:
        """
        Evaluate ChatGPT on the dataset.
        
        Args:
            questions: List of evaluation questions
            model: ChatGPT model to use
            max_questions: Limit number of questions
            
        Returns:
            ChatbotEvaluation results
        """
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            print("⚠️ OpenAI API not available. Skipping ChatGPT evaluation.")
            return self._create_mock_evaluation("ChatGPT (Not Available)", questions[:max_questions or len(questions)])
            
        if max_questions:
            questions = questions[:max_questions]
            
        print(f"🔄 Evaluating ChatGPT ({model}) on {len(questions)} questions...")
        
        results = []
        correct = 0
        total_time = 0
        
        for i, q in enumerate(questions):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(questions)}")
                
            start_time = time.time()
            
            try:
                prompt = self._format_question_for_api(q)
                
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia về K-pop. Trả lời ngắn gọn."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                
                answer_text = response.choices[0].message.content.strip()
                predicted = self._parse_api_response(answer_text, q['question_type'], q.get('choices', []))
                confidence = 0.8
                
            except Exception as e:
                predicted = "Error"
                confidence = 0.0
                
            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time
            
            is_correct = predicted == q['answer']
            if is_correct:
                correct += 1
                
            results.append(EvaluationResult(
                question_id=q['id'],
                question=q['question'],
                question_type=q['question_type'],
                correct_answer=q['answer'],
                predicted_answer=predicted,
                is_correct=is_correct,
                confidence=confidence,
                response_time=response_time,
                hops=q['hops'],
                category=q['category']
            ))
            
            # Rate limiting
            time.sleep(0.5)
            
        # Calculate metrics (same as above)
        accuracy = correct / len(questions) if questions else 0
        
        accuracy_by_hops = {}
        for hop in [1, 2, 3]:
            hop_results = [r for r in results if r.hops == hop]
            if hop_results:
                accuracy_by_hops[hop] = sum(1 for r in hop_results if r.is_correct) / len(hop_results)
                
        accuracy_by_type = {}
        for qtype in ['true_false', 'yes_no', 'multiple_choice']:
            type_results = [r for r in results if r.question_type == qtype]
            if type_results:
                accuracy_by_type[qtype] = sum(1 for r in type_results if r.is_correct) / len(type_results)
                
        accuracy_by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                accuracy_by_category[cat] = sum(1 for r in cat_results if r.is_correct) / len(cat_results)
                
        return ChatbotEvaluation(
            chatbot_name=f"ChatGPT ({model})",
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            accuracy_by_hops=accuracy_by_hops,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_category=accuracy_by_category,
            avg_response_time=total_time / len(questions) if questions else 0,
            evaluated_at=datetime.now().isoformat(),
            results=results
        )
    
    def evaluate_gemini(
        self,
        questions: List[Dict],
        max_questions: Optional[int] = None,
        model_name: str = "gemini-pro"
    ) -> ChatbotEvaluation:
        """
        Evaluate Google Gemini chatbot.
        
        Args:
            questions: List of evaluation questions
            max_questions: Limit number of questions
            model_name: Gemini model to use
            
        Returns:
            ChatbotEvaluation results
        """
        if not GEMINI_AVAILABLE:
            print("⚠️ Google Generative AI library not installed.")
            print("   Install with: pip install google-generativeai")
            print("   ⚠️ Returning 0% accuracy (mock evaluation)")
            return self._create_mock_evaluation("Gemini (Not Available - Library Missing)", questions)
        
        if not self.google_api_key:
            print("⚠️ Google API key not found. Set GOOGLE_API_KEY env var or pass google_api_key parameter")
            print("   ⚠️ Returning 0% accuracy (mock evaluation)")
            return self._create_mock_evaluation("Gemini (No API Key)", questions)
        
        if max_questions:
            questions = questions[:max_questions]
        
        print(f"🔄 Evaluating Gemini ({model_name}) on {len(questions)} questions...")
        
        # Initialize model
        if not self.gemini_model:
            genai.configure(api_key=self.google_api_key)
            try:
                # Try the requested model first
                self.gemini_model = genai.GenerativeModel(model_name)
            except Exception as e:
                # List available models and use one
                try:
                    models = genai.list_models()
                    available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                    
                    # Try preferred models first
                    preferred = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.0-flash-exp']
                    for pref in preferred:
                        matching = [m for m in available if pref in m]
                        if matching:
                            try:
                                self.gemini_model = genai.GenerativeModel(matching[0])
                                print(f"✅ Using {matching[0]} instead")
                                break
                            except:
                                continue
                    
                    # If still None, use first available
                    if not self.gemini_model and available:
                        self.gemini_model = genai.GenerativeModel(available[0])
                        print(f"✅ Using available model: {available[0]}")
                    else:
                        raise Exception(f"Could not initialize any Gemini model: {e}")
                except Exception as e2:
                    raise Exception(f"Could not initialize any Gemini model: {e2}")
        
        results = []
        correct = 0
        total_time = 0
        
        for i, q in enumerate(questions):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(questions)}")
            
            start_time = time.time()
            
            try:
                prompt = self._format_question_for_api(q)
                
                response = self.gemini_model.generate_content(prompt)
                
                # Handle different response formats
                if hasattr(response, 'text'):
                    answer_text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    answer_text = response.candidates[0].content.parts[0].text.strip()
                else:
                    answer_text = str(response).strip()
                
                predicted = self._parse_api_response(answer_text, q['question_type'], q.get('choices', []))
                confidence = 0.8
                
                # Debug first few responses
                if i < 3:
                    print(f"    📝 Q{i+1}: {q['question'][:50]}...")
                    print(f"       Expected: {q['answer']}, Got: {predicted}, Raw: {answer_text[:100]}")
                
            except Exception as e:
                predicted = "Error"
                confidence = 0.0
                error_msg = str(e)
                
                # Check for quota/billing errors
                if "quota" in error_msg.lower() or "429" in error_msg or "billing" in error_msg.lower() or "ResourceExhausted" in error_msg:
                    print(f"    ⚠️ Quota exceeded on question {i+1}")
                    if i == 0:
                        print(f"    ⚠️ Gemini API quota has been exceeded.")
                        print(f"    💡 Solutions:")
                        print(f"       1. Wait and retry later (rate limit resets)")
                        print(f"       2. Check billing: https://ai.dev/usage?tab=rate-limit")
                        print(f"       3. Upgrade API plan if needed")
                        print(f"       4. Skip Gemini comparison for now")
                    # Break early if quota exceeded
                    if i == 0:
                        print(f"    ⚠️ Stopping Gemini evaluation due to quota limit")
                        break
                else:
                    print(f"    ⚠️ Error on question {i+1}: {error_msg[:100]}")
                    # Debug first error in detail
                    if i == 0:
                        import traceback
                        print(f"    Full traceback:")
                        traceback.print_exc()
            
            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time
            
            is_correct = predicted == q['answer']
            if is_correct:
                correct += 1
                
            results.append(EvaluationResult(
                question_id=q['id'],
                question=q['question'],
                question_type=q['question_type'],
                correct_answer=q['answer'],
                predicted_answer=predicted,
                is_correct=is_correct,
                confidence=confidence,
                response_time=response_time,
                hops=q['hops'],
                category=q['category']
            ))
            
            # Rate limiting
            time.sleep(0.5)
        
        # If no questions were processed due to quota, return mock evaluation
        if not results:
            print(f"    ⚠️ No questions processed due to quota limit")
            return self._create_mock_evaluation("Gemini (Quota Exceeded)", questions)
        
        accuracy = correct / len(results) if results else 0
        
        # Calculate metrics
        accuracy_by_hops = {}
        for hop in [1, 2, 3]:
            hop_results = [r for r in results if r.hops == hop]
            if hop_results:
                accuracy_by_hops[hop] = sum(1 for r in hop_results if r.is_correct) / len(hop_results)
        
        accuracy_by_type = {}
        for qtype in ['true_false', 'yes_no', 'multiple_choice']:
            type_results = [r for r in results if r.question_type == qtype]
            if type_results:
                accuracy_by_type[qtype] = sum(1 for r in type_results if r.is_correct) / len(type_results)
        
        accuracy_by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                accuracy_by_category[cat] = sum(1 for r in cat_results if r.is_correct) / len(cat_results)
        
        return ChatbotEvaluation(
            chatbot_name=f"Gemini ({model_name})",
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            accuracy_by_hops=accuracy_by_hops,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_category=accuracy_by_category,
            avg_response_time=total_time / len(questions) if questions else 0,
            evaluated_at=datetime.now().isoformat(),
            results=results
        )
        
    def evaluate_baseline(
        self,
        questions: List[Dict],
        max_questions: Optional[int] = None
    ) -> ChatbotEvaluation:
        """
        Evaluate random baseline (for comparison).
        
        Args:
            questions: List of evaluation questions
            max_questions: Limit number of questions
            
        Returns:
            ChatbotEvaluation results
        """
        import random
        
        if max_questions:
            questions = questions[:max_questions]
            
        print(f"🔄 Evaluating Random Baseline on {len(questions)} questions...")
        
        results = []
        correct = 0
        
        for q in questions:
            if q['question_type'] == 'true_false':
                predicted = random.choice(["Đúng", "Sai"])
            elif q['question_type'] == 'yes_no':
                predicted = random.choice(["Có", "Không"])
            else:
                predicted = random.choice(['A', 'B', 'C', 'D'])
                
            is_correct = predicted == q['answer']
            if is_correct:
                correct += 1
                
            results.append(EvaluationResult(
                question_id=q['id'],
                question=q['question'],
                question_type=q['question_type'],
                correct_answer=q['answer'],
                predicted_answer=predicted,
                is_correct=is_correct,
                confidence=0.25 if q['question_type'] == 'multiple_choice' else 0.5,
                response_time=0.001,
                hops=q['hops'],
                category=q['category']
            ))
            
        accuracy = correct / len(questions) if questions else 0
        
        # Calculate by hops
        accuracy_by_hops = {}
        for hop in [1, 2, 3]:
            hop_results = [r for r in results if r.hops == hop]
            if hop_results:
                accuracy_by_hops[hop] = sum(1 for r in hop_results if r.is_correct) / len(hop_results)
                
        return ChatbotEvaluation(
            chatbot_name="Random Baseline",
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            accuracy_by_hops=accuracy_by_hops,
            accuracy_by_type={},
            accuracy_by_category={},
            avg_response_time=0.001,
            evaluated_at=datetime.now().isoformat(),
            results=results
        )
        
    def _format_question_for_api(self, question: Dict) -> str:
        """Format question for API call."""
        q = question['question']
        
        if question['question_type'] == 'true_false':
            return f"Nhận định sau đúng hay sai? Chỉ trả lời 'Đúng' hoặc 'Sai'.\n\n{q}"
        elif question['question_type'] == 'yes_no':
            return f"Trả lời 'Có' hoặc 'Không': {q}"
        else:
            choices = question['choices']
            choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            return f"Chọn đáp án đúng (trả lời A, B, C, hoặc D):\n\n{q}\n\n{choices_str}"
            
    def _parse_api_response(
        self,
        response: str,
        question_type: str,
        choices: List[str] = None
    ) -> str:
        """Parse API response to standard format."""
        if not response:
            return "Error"
            
        response_upper = response.upper().strip()
        response_lower = response.lower().strip()
        
        if question_type == 'true_false':
            # More flexible matching
            if any(word in response_upper for word in ['ĐÚNG', 'TRUE', 'CORRECT', 'ĐỒNG Ý', 'CHÍNH XÁC']):
                return "Đúng"
            elif any(word in response_upper for word in ['SAI', 'FALSE', 'INCORRECT', 'KHÔNG ĐÚNG', 'KHÔNG']):
                return "Sai"
            else:
                # Default based on first word
                first_word = response_upper.split()[0] if response_upper.split() else ""
                if first_word in ['ĐÚNG', 'TRUE', 'YES']:
                    return "Đúng"
                else:
                    return "Sai"
        elif question_type == 'yes_no':
            # More flexible matching for yes/no
            if any(word in response_upper for word in ['CÓ', 'YES', 'ĐÚNG', 'TRUE', 'CHÍNH XÁC']):
                return "Có"
            elif any(word in response_upper for word in ['KHÔNG', 'NO', 'FALSE', 'SAI', 'INCORRECT']):
                return "Không"
            else:
                # Check first word or first sentence
                first_word = response_upper.split()[0] if response_upper.split() else ""
                if first_word in ['CÓ', 'YES', 'ĐÚNG']:
                    return "Có"
                else:
                    return "Không"
        else:
            # Multiple choice - look for letter
            for letter in ['A', 'B', 'C', 'D', 'E']:
                # Check if letter appears as standalone or in pattern like "A." or "A)"
                if letter in response_upper:
                    # Make sure it's not part of a word
                    import re
                    pattern = r'\b' + letter + r'\b'
                    if re.search(pattern, response_upper):
                        return letter
            # If no letter found, return first choice as default
            return "A"
            
    def _create_mock_evaluation(
        self,
        name: str,
        questions: List[Dict]
    ) -> ChatbotEvaluation:
        """Create mock evaluation when API not available."""
        return ChatbotEvaluation(
            chatbot_name=name,
            total_questions=len(questions),
            correct_answers=0,
            accuracy=0.0,
            accuracy_by_hops={},
            accuracy_by_type={},
            accuracy_by_category={},
            avg_response_time=0.0,
            evaluated_at=datetime.now().isoformat(),
            results=[]
        )
        
    def compare_chatbots(
        self,
        questions: List[Dict],
        include_chatgpt: bool = False,
        include_gemini: bool = False,
        include_baseline: bool = True,
        max_questions: Optional[int] = None,
        output_path: str = "data/comparison_results.json"
    ) -> Dict:
        """
        Run full comparison between chatbots.
        
        Args:
            questions: Evaluation dataset
            include_chatgpt: Include ChatGPT in comparison
            include_baseline: Include random baseline
            max_questions: Limit questions per chatbot
            output_path: Path to save results
            
        Returns:
            Comparison results dictionary
        """
        if max_questions:
            questions = questions[:max_questions]
            
        print(f"\n{'='*60}")
        print(f"🔬 Chatbot Comparison ({len(questions)} questions)")
        print(f"{'='*60}\n")
        
        evaluations = {}
        
        # 1. K-pop Chatbot
        if self.kpop_chatbot:
            eval_kpop = self.evaluate_kpop_chatbot(questions)
            evaluations['kpop_chatbot'] = eval_kpop
            print(f"✅ K-pop Chatbot: {eval_kpop.accuracy:.2%} accuracy")
            
        # 2. ChatGPT
        if include_chatgpt:
            eval_chatgpt = self.evaluate_chatgpt(questions)
            evaluations['chatgpt'] = eval_chatgpt
            print(f"✅ ChatGPT: {eval_chatgpt.accuracy:.2%} accuracy")
        
        # 3. Gemini
        if include_gemini:
            try:
                eval_gemini = self.evaluate_gemini(questions)
                evaluations['gemini'] = eval_gemini
                if "Quota Exceeded" in eval_gemini.chatbot_name:
                    print(f"⚠️ Gemini: Quota exceeded (skipped)")
                else:
                    print(f"✅ Gemini: {eval_gemini.accuracy:.2%} accuracy")
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e) or "ResourceExhausted" in str(e):
                    print(f"⚠️ Gemini: Quota exceeded (skipped)")
                    evaluations['gemini'] = self._create_mock_evaluation("Gemini (Quota Exceeded)", questions)
                else:
                    print(f"⚠️ Gemini: Error - {str(e)[:100]}")
                    evaluations['gemini'] = self._create_mock_evaluation("Gemini (Error)", questions)
            
        # 4. Baseline
        if include_baseline:
            eval_baseline = self.evaluate_baseline(questions)
            evaluations['baseline'] = eval_baseline
            print(f"✅ Baseline: {eval_baseline.accuracy:.2%} accuracy")
            
        # Create comparison report
        comparison = {
            "metadata": {
                "total_questions": len(questions),
                "comparison_date": datetime.now().isoformat(),
                "chatbots_compared": list(evaluations.keys())
            },
            "summary": {},
            "detailed_results": {}
        }
        
        for name, eval_result in evaluations.items():
            comparison["summary"][name] = {
                "accuracy": eval_result.accuracy,
                "accuracy_by_hops": eval_result.accuracy_by_hops,
                "accuracy_by_type": eval_result.accuracy_by_type,
                "avg_response_time": eval_result.avg_response_time
            }
            comparison["detailed_results"][name] = {
                "correct_answers": eval_result.correct_answers,
                "total_questions": eval_result.total_questions,
                "accuracy_by_category": eval_result.accuracy_by_category,
                "results": [asdict(r) for r in eval_result.results[:100]]  # Sample
            }
            
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
            
        print(f"\n📄 Results saved to: {output_path}")
        
        # Print comparison table
        self._print_comparison_table(evaluations)
        
        return comparison
        
    def _print_comparison_table(self, evaluations: Dict[str, ChatbotEvaluation]):
        """Print formatted comparison table."""
        print(f"\n{'='*70}")
        print(f"{'📊 COMPARISON RESULTS':^70}")
        print(f"{'='*70}")
        
        # Header
        print(f"{'Chatbot':<30} {'Accuracy':>10} {'1-hop':>8} {'2-hop':>8} {'3-hop':>8}")
        print("-" * 70)
        
        # Data rows
        for name, eval_result in evaluations.items():
            hop1 = eval_result.accuracy_by_hops.get(1, 0)
            hop2 = eval_result.accuracy_by_hops.get(2, 0)
            hop3 = eval_result.accuracy_by_hops.get(3, 0)
            
            print(f"{name:<30} {eval_result.accuracy:>9.1%} {hop1:>7.1%} {hop2:>7.1%} {hop3:>7.1%}")
            
        print("=" * 70)
        
        # Winner
        if evaluations:
            winner = max(evaluations.items(), key=lambda x: x[1].accuracy)
            print(f"\n🏆 Best performer: {winner[0]} ({winner[1].accuracy:.1%} accuracy)")


def main():
    """Run chatbot comparison."""
    from .chatbot import KpopChatbot
    from .evaluation import EvaluationDatasetGenerator
    
    # Generate evaluation dataset if not exists
    dataset_path = "data/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print("📝 Generating evaluation dataset...")
        generator = EvaluationDatasetGenerator()
        generator.generate_full_dataset(output_path=dataset_path)
        
    # Initialize chatbot
    print("\n🤖 Initializing K-pop Chatbot...")
    chatbot = KpopChatbot(verbose=True)
    
    # Run comparison
    comparison = ChatbotComparison(kpop_chatbot=chatbot)
    questions = comparison.load_evaluation_dataset(dataset_path)
    
    results = comparison.compare_chatbots(
        questions,
        include_chatgpt=False,  # Set True if you have API key
        include_baseline=True,
        max_questions=500  # Limit for testing
    )
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

