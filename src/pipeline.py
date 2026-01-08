"""
Main pipeline for narrative consistency checking.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import pandas as pd
from tqdm import tqdm

from src.config import get_config
from src.llm_providers import create_llm_provider
from src.pathway_ingestion import PathwayVectorStore, Reranker
from src.self_consistency import SelfConsistencyEngine
from src.multi_agent import MultiAgentSystem
from src.ensemble import MultiModelEnsemble, EnsembleVoter


class NarrativeConsistencyChecker:
    """Main pipeline for checking narrative-backstory consistency."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the consistency checker."""
        logger.info("Initializing NarrativeConsistencyChecker")
        
        # Load configuration
        self.config = get_config(config_path)
        
        # Initialize components
        self._init_llm()
        self._init_vector_store()
        self._init_reranker()
        self._init_reasoning_engines()
        
        logger.info("NarrativeConsistencyChecker initialized successfully")
    
    def _init_llm(self):
        """Initialize primary LLM provider."""
        provider_name = self.config.primary_provider
        provider_config = self.config.get_provider_config(provider_name)
        
        self.primary_llm = create_llm_provider(provider_name, provider_config)
        logger.info(f"Initialized primary LLM: {provider_name}")
    
    def _init_vector_store(self):
        """Initialize Pathway vector store."""
        self.vector_store = PathwayVectorStore(self.config._config)
        logger.info("Initialized Pathway vector store")
    
    def _init_reranker(self):
        """Initialize reranker if enabled."""
        reranker_config = self.config.get_reranker_config()
        
        if reranker_config.get('enabled', True):
            self.reranker = Reranker(reranker_config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))
            logger.info("Initialized reranker")
        else:
            self.reranker = None
            logger.info("Reranker disabled")
    
    def _init_reasoning_engines(self):
        """Initialize reasoning engines."""
        # Self-consistency engine
        sc_config = self.config.get_self_consistency_config()
        if sc_config.get('enabled', True):
            self.self_consistency = SelfConsistencyEngine(
                self.primary_llm,
                num_chains=sc_config.get('num_chains', 10),
                voting_strategy=sc_config.get('voting_strategy', 'weighted')
            )
            logger.info("Initialized self-consistency engine")
        else:
            self.self_consistency = None
        
        # Multi-agent system
        ma_config = self.config.get_multi_agent_config()
        if ma_config.get('enabled', True):
            self.multi_agent = MultiAgentSystem(self.primary_llm, ma_config)
            logger.info("Initialized multi-agent system")
        else:
            self.multi_agent = None
        
        # Ensemble system
        ensemble_config = self.config.get_ensemble_config()
        if ensemble_config.get('enabled', True) and len(ensemble_config.get('models', [])) > 1:
            self.ensemble = MultiModelEnsemble(ensemble_config)
            logger.info("Initialized ensemble system")
        else:
            self.ensemble = None
    
    def process_single_example(self, narrative_text: str, backstory: str, 
                              narrative_id: str = "narrative") -> Dict[str, Any]:
        """
        Process a single narrative-backstory pair.
        
        Args:
            narrative_text: Full text of the narrative
            backstory: Hypothetical backstory
            narrative_id: Unique identifier for this narrative
        
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        logger.info(f"Processing narrative {narrative_id}")
        
        # Step 1: Ingest narrative and create vector store
        logger.info("Step 1: Ingesting narrative into vector store")
        chunking_strategy = self.config.get('pathway', {}).get('chunking', {}).get('strategy', 'semantic')
        chunks = self.vector_store.ingest_narrative(narrative_text, narrative_id, strategy=chunking_strategy)
        
        # Step 2: Extract key queries from backstory
        logger.info("Step 2: Extracting key queries from backstory")
        queries = self._extract_queries_from_backstory(backstory)
        
        # Step 3: Retrieve relevant evidence
        logger.info("Step 3: Retrieving relevant evidence")
        evidence = self._retrieve_evidence(queries, chunks)
        
        # Step 4: Rerank evidence if enabled
        if self.reranker:
            logger.info("Step 4: Reranking evidence")
            evidence = self._rerank_evidence(backstory, evidence)
        
        # Step 5: Run reasoning engines
        logger.info("Step 5: Running reasoning engines")
        predictions = []
        
        # Self-consistency
        if self.self_consistency:
            logger.info("Running self-consistency reasoning")
            chains = self.self_consistency.generate_reasoning_chains(
                narrative_text[:5000],  # Use first 5k chars as summary
                backstory,
                evidence
            )
            sc_result = self.self_consistency.aggregate_chains(chains)
            predictions.append({
                'decision': sc_result['decision'],
                'confidence': sc_result['confidence'],
                'reasoning': sc_result['reasoning'],
                'weight': 1.5,  # Higher weight for self-consistency
                'method': 'self_consistency'
            })
        
        # Multi-agent
        if self.multi_agent:
            logger.info("Running multi-agent deliberation")
            ma_result = self.multi_agent.deliberate(backstory, evidence)
            predictions.append({
                'decision': ma_result['decision'],
                'confidence': ma_result['confidence'],
                'reasoning': ma_result['reasoning'],
                'weight': 1.2,
                'method': 'multi_agent'
            })
        
        # Ensemble (if no other methods available)
        if not predictions and self.ensemble:
            logger.info("Running ensemble prediction")
            # Create a prompt for ensemble
            prompt = self._create_consistency_prompt(backstory, evidence)
            ensemble_result = self.ensemble.ensemble_predict(prompt)
            predictions.append({
                'decision': ensemble_result['decision'],
                'confidence': ensemble_result['confidence'],
                'reasoning': ensemble_result['reasoning'],
                'weight': 1.0,
                'method': 'ensemble'
            })
        
        # Fallback: single model prediction
        if not predictions:
            logger.info("Running single model prediction (fallback)")
            prompt = self._create_consistency_prompt(backstory, evidence)
            response = self.primary_llm.generate(prompt)
            parsed = self._parse_llm_response(response)
            predictions.append({
                'decision': parsed['decision'],
                'confidence': parsed['confidence'],
                'reasoning': parsed['reasoning'],
                'weight': 1.0,
                'method': 'single_model'
            })
        
        # Step 6: Aggregate predictions
        logger.info("Step 6: Aggregating predictions")
        voter = EnsembleVoter(self.config.get_ensemble_config())
        final_result = voter.vote(predictions)
        
        # Enhanced logging with clear verdict
        verdict = "✓ CONSISTENT" if final_result['decision'] == 1 else "✗ INCONSISTENT"
        logger.info("=" * 80)
        logger.info(f"FINAL VERDICT for Example {narrative_id}: {verdict}")
        logger.info(f"Confidence: {final_result['confidence']:.1%}")
        logger.info(f"Reasoning: {final_result['reasoning'][:150]}...")
        logger.info("=" * 80)
        
        return {
            'narrative_id': narrative_id,
            'decision': final_result['decision'],
            'confidence': final_result['confidence'],
            'reasoning': final_result['reasoning'],
            'predictions': predictions,
            'evidence_count': len(evidence)
        }
    
    def _extract_queries_from_backstory(self, backstory: str) -> List[str]:
        """Extract key queries from backstory for evidence retrieval."""
        # Use LLM to extract key claims
        prompt = f"""Extract 5-10 key factual claims from this backstory that should be verified against the narrative.
Focus on claims about:
1. Character attributes (name, age, appearance, personality)
2. Past events and experiences
3. Relationships with other characters
4. Beliefs, motivations, fears
5. Skills or abilities

BACKSTORY:
{backstory}

Output each claim as a short query (one per line):"""

        try:
            response = self.primary_llm.generate(prompt, temperature=0.0)
            queries = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            # Add the full backstory as a query too
            queries.append(backstory)
            
            logger.info(f"Extracted {len(queries)} queries from backstory")
            return queries
        except Exception as e:
            logger.error(f"Error extracting queries: {e}")
            return [backstory]
    
    def _retrieve_evidence(self, queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve relevant evidence for all queries."""
        evidence_config = self.config.get_evidence_config()
        max_passages = evidence_config.get('max_passages', 30)
        
        all_evidence = []
        seen_chunks = set()
        
        for query in queries:
            # Hybrid search for each query
            results = self.vector_store.hybrid_search(query, chunks, top_k=10)
            
            for result in results:
                chunk_id = result['chunk_id']
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_evidence.append(result)
        
        # Sort by score and take top passages
        all_evidence.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        all_evidence = all_evidence[:max_passages]
        
        logger.info(f"Retrieved {len(all_evidence)} unique evidence passages")
        return all_evidence
    
    def _rerank_evidence(self, backstory: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank evidence using cross-encoder."""
        reranker_config = self.config.get_reranker_config()
        final_k = reranker_config.get('final_k', 20)
        
        reranked = self.reranker.rerank(backstory, evidence, top_k=final_k)
        logger.info(f"Reranked to top {len(reranked)} passages")
        return reranked
    
    def _create_consistency_prompt(self, backstory: str, evidence: List[Dict[str, Any]]) -> str:
        """Create a prompt for consistency checking."""
        evidence_text = "\n\n".join([
            f"[Passage {i+1}]\n{ev['text']}"
            for i, ev in enumerate(evidence[:15])
        ])
        
        return f"""Determine if the following character backstory is CONSISTENT or INCONSISTENT with the narrative evidence.

BACKSTORY:
{backstory}

NARRATIVE EVIDENCE:
{evidence_text}

Analyze carefully and respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation]"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response."""
        decision = 1
        confidence = 0.5
        reasoning = ""
        
        for line in response.split('\n'):
            line_stripped = line.strip()
            
            if line_stripped.startswith('DECISION:'):
                decision_text = line_stripped.split(':', 1)[1].strip().upper()
                if 'INCONSISTENT' in decision_text:
                    decision = 0
                elif 'CONSISTENT' in decision_text:
                    decision = 1
            
            elif line_stripped.startswith('CONFIDENCE:'):
                try:
                    conf_text = line_stripped.split(':', 1)[1].strip()
                    confidence = float(conf_text.split()[0])
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
            
            elif line_stripped.startswith('REASONING:'):
                reasoning = line_stripped.split(':', 1)[1].strip()
        
        if not reasoning:
            reasoning = response[:200]
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def process_dataset(self, dataset_path: str, output_path: str = "results.csv") -> pd.DataFrame:
        """
        Process entire dataset and generate results.
        
        Args:
            dataset_path: Path to dataset directory or file
            output_path: Path to save results CSV
        
        Returns:
            DataFrame with results
        """
        logger.info(f"Processing dataset from {dataset_path}")
        
        # Load dataset (implementation depends on actual dataset format)
        examples = self._load_dataset(dataset_path)
        
        results = []
        
        for example in tqdm(examples, desc="Processing examples"):
            try:
                result = self.process_single_example(
                    narrative_text=example['narrative'],
                    backstory=example['backstory'],
                    narrative_id=example['id']
                )
                
                results.append({
                    'Story ID': example['id'],
                    'Prediction': result['decision'],
                    'Rationale': result['reasoning'][:200]  # Limit rationale length
                })
                
            except Exception as e:
                logger.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
                results.append({
                    'Story ID': example.get('id', 'unknown'),
                    'Prediction': 1,
                    'Rationale': f"Error: {str(e)}"
                })
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        return df
    
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from CSV file with book narratives."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        path = Path(dataset_path)
        
        # Handle CSV file (train.csv or test.csv)
        if path.is_file() and path.suffix == '.csv':
            logger.info(f"Loading CSV dataset: {path}")
            df = pd.read_csv(path)
            
            # Load narrative texts
            narratives = {}
            data_dir = path.parent / 'data'
            
            # Map book names to file names
            book_files = {
                'In Search of the Castaways': 'In search of the castaways.txt',
                'The Count of Monte Cristo': 'The Count of Monte Cristo.txt'
            }
            
            for book_name, filename in book_files.items():
                book_path = data_dir / filename
                if book_path.exists():
                    logger.info(f"Loading narrative: {book_name}")
                    with open(book_path, 'r', encoding='utf-8') as f:
                        narratives[book_name] = f.read()
                else:
                    logger.warning(f"Narrative file not found: {book_path}")
            
            # Create examples
            examples = []
            for _, row in df.iterrows():
                book_name = row['book_name']
                if book_name in narratives:
                    examples.append({
                        'id': str(row['id']),
                        'narrative': narratives[book_name],
                        'backstory': row['content'],
                        'book_name': book_name,
                        'character': row.get('char', ''),
                        'true_label': row.get('label', None)  # None for test.csv
                    })
                else:
                    logger.error(f"No narrative found for book: {book_name}")
            
            logger.info(f"Loaded {len(examples)} examples from CSV")
            return examples
        
        # Handle JSON/JSONL format
        elif path.is_file():
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
            elif path.suffix == '.jsonl':
                examples = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        examples.append(json.loads(line))
                return examples
        
        raise ValueError(f"Invalid dataset path: {dataset_path}. Expected CSV, JSON, or JSONL file.")
