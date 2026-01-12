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
        """Initialize primary LLM provider with triple fallback."""
        from src.llm_providers import FallbackProvider
        
        provider_name = self.config.primary_provider
        provider_config = self.config.get_provider_config(provider_name)
        
        primary = create_llm_provider(provider_name, provider_config)
        
        # Setup triple fallback: HuggingFace → Groq → Ollama (per-request)
        if provider_name.lower() == 'huggingface':
            try:
                # Second: Groq
                groq_config = self.config.get_provider_config('groq')
                fallback = create_llm_provider('groq', groq_config)
                
                # Third: Ollama
                try:
                    ollama_config = self.config.get_provider_config('ollama')
                    secondary_fallback = create_llm_provider('ollama', ollama_config)
                    self.primary_llm = FallbackProvider(
                        primary=primary, 
                        fallback=fallback, 
                        secondary_fallback=secondary_fallback
                    )
                    logger.info("✓ Triple fallback: HuggingFace → Groq → Ollama (per-request)")
                except Exception as e:
                    self.primary_llm = FallbackProvider(primary=primary, fallback=fallback)
                    logger.info("✓ Double fallback: HuggingFace → Groq (per-request)")
            except Exception as e:
                logger.warning(f"Could not initialize fallback chain: {e}. Using HuggingFace only.")
                self.primary_llm = primary
        elif provider_name.lower() in ['groq', 'deepseek']:
            try:
                ollama_config = self.config.get_provider_config('ollama')
                fallback = create_llm_provider('ollama', ollama_config)
                self.primary_llm = FallbackProvider(primary=primary, fallback=fallback)
                logger.info(f"✓ Double fallback: {provider_name} → Ollama (per-request)")
            except Exception as e:
                logger.warning(f"Could not initialize Ollama fallback: {e}. Using {provider_name} only.")
                self.primary_llm = primary
        else:
            self.primary_llm = primary
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
            evidence_cfg = self.config.get_evidence_config()
            max_evidence_passages = int(evidence_cfg.get('llm_passages', evidence_cfg.get('max_passages', 15)))
            early_stop_conf = float(sc_config.get('early_stop_confidence', 0.85))
            self.self_consistency = SelfConsistencyEngine(
                self.primary_llm,
                num_chains=sc_config.get('num_chains', 10),
                voting_strategy=sc_config.get('voting_strategy', 'weighted'),
                max_evidence_passages=max_evidence_passages,
                early_stop_confidence=early_stop_conf,
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
    
    def process_single_example(
        self,
        narrative_text: str,
        backstory: str,
        story_id: str = "story",
        narrative_id: str = "narrative",
    ) -> Dict[str, Any]:
        """
        Process a single narrative-backstory pair.
        
        Args:
            narrative_text: Full text of the narrative
            backstory: Hypothetical backstory
            narrative_id: Unique identifier for this narrative
        
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        logger.info(f"Processing story {story_id} (narrative: {narrative_id})")
        
        # Step 1: Ingest narrative and create vector store
        logger.info("Step 1: Ingesting narrative into vector store")
        chunking_strategy = self.config.get('pathway', {}).get('chunking', {}).get('strategy', 'semantic')
        chunks = self.vector_store.ingest_narrative(narrative_text, narrative_id, strategy=chunking_strategy)
        
        # Step 2: Extract key queries from backstory
        logger.info("Step 2: Extracting key queries from backstory")
        queries = self._extract_queries_from_backstory(backstory)

        # Use extracted queries as high-signal claims for reasoning (token efficient)
        claims_text = "\n".join(f"- {q}" for q in queries[:3])
        backstory_for_llm = f"KEY CLAIMS:\n{claims_text}\n\nBACKSTORY:\n{backstory}"
        
        # Step 3: Retrieve relevant evidence
        logger.info("Step 3: Retrieving relevant evidence")
        evidence = self._retrieve_evidence(queries, chunks)
        
        # Step 4: Rerank evidence if enabled
        if self.reranker:
            logger.info("Step 4: Reranking evidence")
            evidence = self._rerank_evidence(backstory, evidence)

        # Token optimization: truncate evidence text before sending to LLM
        evidence_cfg = self.config.get_evidence_config()
        context_window = int(evidence_cfg.get('context_window', 500))
        for ev in evidence:
            text = ev.get('text', '')
            if isinstance(text, str) and len(text) > context_window:
                ev['text'] = text[:context_window]
        
        # Step 5: Run reasoning engines
        logger.info("Step 5: Running reasoning engines")
        predictions = []
        
        # Self-consistency
        if self.self_consistency:
            logger.info("Running self-consistency reasoning")
            chains = self.self_consistency.generate_reasoning_chains(
                narrative_text[:5000],  # Use first 5k chars as summary
                backstory_for_llm,
                evidence
            )
            sc_result = self.self_consistency.aggregate_chains(chains)
            
            # Only use result if we got a valid decision
            if sc_result.get('decision') is not None:
                predictions.append({
                    'decision': sc_result['decision'],
                    'confidence': sc_result['confidence'],
                    'reasoning': sc_result['reasoning'],
                    'weight': 1.5,  # Higher weight for self-consistency
                    'method': 'self_consistency'
                })
            else:
                logger.warning("Self-consistency failed to produce valid decision, skipping")
        
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
        logger.info(f"FINAL VERDICT for Example {story_id}: {verdict}")
        logger.info(f"Confidence: {final_result['confidence']:.1%}")
        logger.info(f"Reasoning: {final_result['reasoning'][:150]}...")
        logger.info("=" * 80)
        
        return {
            'story_id': story_id,
            'narrative_id': narrative_id,
            'decision': final_result['decision'],
            'confidence': final_result['confidence'],
            'reasoning': final_result['reasoning'],
            'predictions': predictions,
            'evidence_count': len(evidence)
        }
    
    def _extract_queries_from_backstory(self, backstory: str) -> List[str]:
        """Extract key queries from backstory for evidence retrieval."""
        # Token-optimized: get a few short retrieval queries (saves time and tokens)
        prompt = (
            "Extract up to 3 short search queries (max 8 words each) that would help verify this backstory. "
            "One query per line. No numbering, no extra text.\n\n"
            f"BACKSTORY:\n{backstory[:700]}"
        )

        try:
            response = self.primary_llm.generate(prompt, temperature=0.0, max_tokens=80)
            raw_lines = [line.strip() for line in (response or "").split('\n')]
            queries = [ln for ln in raw_lines if ln and not ln.startswith('#')]
            queries = queries[:3]
            if not queries:
                return [backstory]
            logger.info(f"Extracted {len(queries)} queries from backstory")
            return queries
        except Exception as e:
            logger.error(f"Error extracting queries: {e}")
            return [backstory]
    
    def _retrieve_evidence(self, queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve relevant evidence for all queries."""
        evidence_config = self.config.get_evidence_config()
        max_passages = evidence_config.get('max_passages', 30)

        # If reranker is enabled, retrieve more candidates first, then rerank down.
        reranker_config = self.config.get_reranker_config()
        if self.reranker:
            candidate_k = int(reranker_config.get('top_k', 50))
        else:
            candidate_k = int(max_passages)

        num_queries = max(1, len(queries))
        per_query_top_k = max(10, min(25, (candidate_k + num_queries - 1) // num_queries))
        
        all_evidence = []
        seen_chunks = set()
        
        for query in queries:
            # Hybrid search for each query
            results = self.vector_store.hybrid_search(query, chunks, top_k=per_query_top_k)
            
            for result in results:
                chunk_id = result['chunk_id']
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_evidence.append(result)
        
        # Sort by score and take top passages
        all_evidence.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        all_evidence = all_evidence[:candidate_k]
        
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
            
            if line_stripped.upper().startswith('DECISION:'):
                decision_text = line_stripped.split(':', 1)[1].strip().lower()
                if any(t in decision_text for t in ['inconsistent', 'inconistent', 'inconsistant', 'contradict']):
                    decision = 0
                elif 'consistent' in decision_text:
                    decision = 1
            
            elif line_stripped.upper().startswith('CONFIDENCE:'):
                try:
                    conf_text = line_stripped.split(':', 1)[1].strip()
                    confidence = float(conf_text.split()[0])
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
            
            elif line_stripped.upper().startswith('REASONING:'):
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
        
        # Resume support: if output exists, skip already processed Story IDs
        processed_ids = set()
        if Path(output_path).exists():
            try:
                existing = pd.read_csv(output_path)
                if 'Story ID' in existing.columns:
                    processed_ids = set(existing['Story ID'].astype(str).tolist())
                    logger.info(f"Resuming: found {len(processed_ids)} existing rows in {output_path}")
            except Exception as e:
                logger.warning(f"Could not read existing output for resume: {e}")

        results: List[Dict[str, Any]] = []
        
        for example in tqdm(examples, desc="Processing examples"):
            if str(example.get('id')) in processed_ids:
                continue
            try:
                result = self.process_single_example(
                    narrative_text=example['narrative'],
                    backstory=example['backstory'],
                    story_id=example['id'],
                    # Cache and reuse expensive chunking/embeddings per book
                    narrative_id=example.get('book_name', example['id']),
                )
                
                row = {
                    'Story ID': example['id'],
                    'Prediction': result['decision'],
                    'Rationale': result['reasoning'][:200]  # Limit rationale length
                }
                results.append(row)

                # Incremental save so progress isn't lost on OOM/kill
                try:
                    df_one = pd.DataFrame([row])
                    write_header = not Path(output_path).exists() or Path(output_path).stat().st_size == 0
                    df_one.to_csv(output_path, mode='a', index=False, header=write_header)
                except Exception as e:
                    logger.warning(f"Incremental save failed: {e}")
                
            except Exception as e:
                logger.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
                row = {
                    'Story ID': example.get('id', 'unknown'),
                    'Prediction': 1,
                    'Rationale': f"Error: {str(e)}"
                }
                results.append(row)

                try:
                    df_one = pd.DataFrame([row])
                    write_header = not Path(output_path).exists() or Path(output_path).stat().st_size == 0
                    df_one.to_csv(output_path, mode='a', index=False, header=write_header)
                except Exception as e2:
                    logger.warning(f"Incremental save failed: {e2}")
        
        # Save results (final write) - keep for convenience, but file already updated incrementally
        df = pd.DataFrame(results)
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
