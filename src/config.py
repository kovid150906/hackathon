"""
Configuration loader for the Narrative Consistency Checker.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Optional dotenv support (for .env files)
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = lambda: None  # No-op function
    logger.debug("python-dotenv not available, skipping .env file loading")

class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration."""
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        
        # Load environment variables if dotenv available
        if DOTENV_AVAILABLE:
            load_dotenv()
        self._load_config()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        if DOTENV_AVAILABLE:
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)
                logger.info("Loaded environment variables from .env")
            else:
                logger.warning("No .env file found. Using environment variables only.")
        else:
            logger.debug(".env file support not available (python-dotenv not installed)")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_provider_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific LLM provider."""
        if provider is None:
            provider = self.get('llm_provider', 'groq')
        
        provider_config = self.get(f'providers.{provider}', {})
        
        # Add API key from environment
        api_key_map = {
            'groq': 'GROQ_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'ollama': 'OLLAMA_BASE_URL'
        }
        
        env_key = api_key_map.get(provider)
        if env_key:
            env_value = os.getenv(env_key)
            if env_value:
                if provider == 'ollama':
                    provider_config['base_url'] = env_value
                else:
                    provider_config['api_key'] = env_value
        
        return provider_config
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self.get('embeddings', {})
    
    def get_reranker_config(self) -> Dict[str, Any]:
        """Get reranker configuration."""
        return self.get('reranker', {})
    
    def get_pathway_config(self) -> Dict[str, Any]:
        """Get Pathway configuration."""
        return self.get('pathway', {})
    
    def get_self_consistency_config(self) -> Dict[str, Any]:
        """Get self-consistency configuration."""
        return self.get('self_consistency', {})
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """Get ensemble configuration."""
        return self.get('ensemble', {})
    
    def get_multi_agent_config(self) -> Dict[str, Any]:
        """Get multi-agent configuration."""
        return self.get('multi_agent', {})
    
    def get_evidence_config(self) -> Dict[str, Any]:
        """Get evidence extraction configuration."""
        return self.get('evidence', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('output', {})
    
    @property
    def primary_provider(self) -> str:
        """Get primary LLM provider."""
        return self.get('llm_provider', 'groq')


# Global configuration instance
_config: Optional[Config] = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def reload_config(config_path: str = "config.yaml"):
    """Reload configuration."""
    global _config
    _config = Config(config_path)
    return _config
