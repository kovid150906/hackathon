"""
Abstract base class for LLM providers with implementations for multiple providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from loguru import logger

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config
        self.model = config.get('model')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4096)
        self._client = None
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages."""
        pass
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class GroqProvider(LLMProvider):
    """Groq API provider (FREE)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key') or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"Initialized Groq provider with model: {self.model}")
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.generate_with_system("You are a helpful assistant.", prompt, **kwargs)
    
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages with rate limit fallback."""
        import time
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        max_retries = 1  # Only retry once to save time
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    logger.error(f"Groq rate limit exceeded. Please wait or switch to Ollama.")
                    logger.error(f"To use Ollama: python main.py --provider ollama --dataset train.csv")
                    raise Exception("RATE_LIMIT: Switch to Ollama (free, unlimited) or wait 1 hour")
                logger.error(f"Groq API error: {e}")
                raise


class OllamaProvider(LLMProvider):
    """Ollama local provider (FREE)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        # Large context windows can require multiple GiB of KV cache.
        # Default to a conservative value for low-RAM machines.
        self.num_ctx = config.get('num_ctx', 2048)
        
        try:
            import ollama
            self._client = ollama.Client(host=self.base_url)
            # Ensure model is taken from provider config or env override
            env_model = os.getenv('OLLAMA_MODEL')
            if env_model:
                self.model = env_model

            logger.info(f"Initialized Ollama provider with model: {self.model}")
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.generate_with_system("You are a helpful assistant.", prompt, **kwargs)
    
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages."""
        temperature = kwargs.get('temperature', self.temperature)

        # If Ollama reports an out-of-memory error, retry with smaller context.
        candidate_ctx_values = [self.num_ctx, 2048, 1536, 1024]
        seen = set()
        for ctx in candidate_ctx_values:
            if ctx in seen:
                continue
            seen.add(ctx)

            try:
                logger.info(f"Calling Ollama chat with model: {self.model} (num_ctx={ctx})")
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    options={
                        "temperature": temperature,
                        "num_ctx": ctx
                    }
                )
                # Persist the last successful context so future calls use it.
                self.num_ctx = ctx
                return response['message']['content']
            except Exception as e:
                error_str = str(e).lower()
                if 'requires more system memory' in error_str or 'out of memory' in error_str:
                    logger.warning(f"Ollama OOM at num_ctx={ctx}; trying smaller context")
                    last_error = e
                    continue
                logger.error(f"Ollama API error: {e}")
                raise

        logger.error(f"Ollama API error after context retries: {last_error}")
        raise last_error


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (PAID)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic provider with model: {self.model}")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.generate_with_system("You are a helpful assistant.", prompt, **kwargs)
    
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider (PAID)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI provider with model: {self.model}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.generate_with_system("You are a helpful assistant.", prompt, **kwargs)
    
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GoogleProvider(LLMProvider):
    """Google Gemini provider (PAID)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Google provider with model: {self.model}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
    
    def generate_with_system(self, system: str, user: str, **kwargs) -> str:
        """Generate text with system and user messages."""
        # Gemini doesn't have separate system messages, prepend to prompt
        combined_prompt = f"{system}\n\n{user}"
        return self.generate(combined_prompt, **kwargs)


def create_llm_provider(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create LLM provider."""
    providers = {
        'groq': GroqProvider,
        'ollama': OllamaProvider,
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider,
        'google': GoogleProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(config)
