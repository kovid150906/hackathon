"""
Self-consistency reasoning engine with multiple independent chains.
"""
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from loguru import logger
import json
import re
from src.llm_providers import LLMProvider


class SelfConsistencyEngine:
    """Self-consistency reasoning with multiple independent chains."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        num_chains: int = 10,
        voting_strategy: str = "weighted",
        max_evidence_passages: int = 15,
        early_stop_confidence: float = 0.85,
    ):
        """
        Initialize self-consistency engine.
        
        Args:
            llm_provider: LLM provider instance
            num_chains: Number of independent reasoning chains
            voting_strategy: 'majority' or 'weighted'
        """
        self.llm = llm_provider
        self.num_chains = num_chains
        self.voting_strategy = voting_strategy
        self.max_evidence_passages = max_evidence_passages
        self.early_stop_confidence = early_stop_confidence
        logger.info(f"Initialized SelfConsistencyEngine with {num_chains} chains, strategy: {voting_strategy}")
    
    def generate_reasoning_chains(self, narrative: str, backstory: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate multiple independent reasoning chains.
        
        Args:
            narrative: Full narrative text (or summary)
            backstory: Hypothetical backstory
            evidence: Retrieved evidence passages
        
        Returns:
            List of reasoning chains with decisions
        """
        prompts = self._create_varied_prompts(narrative, backstory, evidence)
        chains = []
        
        for i, prompt in enumerate(prompts[:self.num_chains]):
            logger.info(f"Generating reasoning chain {i+1}/{self.num_chains}")
            
            try:
                # Extract content string from prompt dict
                prompt_text = prompt['content'] if isinstance(prompt, dict) else prompt
                temperature = 0.1
                if isinstance(prompt, dict) and 'temperature' in prompt:
                    temperature = prompt['temperature']
                response = self.llm.generate(prompt_text, temperature=temperature)
                parsed = self._parse_response(response)
                chains.append({
                    'chain_id': i,
                    'prompt_type': prompt.get('type', 'unknown') if isinstance(prompt, dict) else 'unknown',
                    'decision': parsed['decision'],
                    'confidence': parsed['confidence'],
                    'reasoning': parsed['reasoning'],
                    'raw_response': response
                })

                # Efficiency win without accuracy loss: early-stop only on very high confidence.
                if (
                    parsed.get('decision') is not None
                    and float(parsed.get('confidence') or 0.0) >= self.early_stop_confidence
                ):
                    logger.info(
                        f"Early-stopping self-consistency at chain {i+1} (confidence={parsed['confidence']:.2f})"
                    )
                    break
            except Exception as e:
                logger.error(f"Error in chain {i}: {e}")
                chains.append({
                    'chain_id': i,
                    'prompt_type': prompt.get('type', 'unknown') if isinstance(prompt, dict) else 'unknown',
                    'decision': None,
                    'confidence': 0.0,
                    'reasoning': f"Error: {str(e)}",
                    'raw_response': ""
                })
        
        return chains
    
    def _create_varied_prompts(self, narrative: str, backstory: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create diverse prompts for different reasoning approaches."""
        
        # Format evidence
        evidence_text = self._format_evidence(evidence, max_passages=self.max_evidence_passages)
        backstory_short = backstory[:1200]
        
        prompts = [
            # Direct analysis
            {
                'type': 'direct',
                'content': f"""You are analyzing whether a character's backstory is consistent with a narrative.

BACKSTORY:
{backstory_short}

EVIDENCE FROM NARRATIVE:
{evidence_text}

Decide if there is any clear contradiction between BACKSTORY and EVIDENCE.

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [1-3 sentences, cite passages]"""
            },

            # Contradiction search (high-signal)
            {
                'type': 'contradiction',
                'content': f"""Find ANY explicit contradiction between backstory and evidence.

BACKSTORY:
{backstory_short}

EVIDENCE:
{evidence_text}

If you find even ONE clear contradiction (names/places/roles/timeline/events) → INCONSISTENT.
If no contradictions found → CONSISTENT.

DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [List contradictions or say none found]"""
            },
            
            # Timeline reconstruction
            {
                'type': 'timeline',
                'content': f"""You are a detective reconstructing a timeline to verify a character's backstory.

PROPOSED BACKSTORY:
{backstory_short}

EVIDENCE FROM NARRATIVE:
{evidence_text}

Task: Build a timeline of events mentioned in both the backstory and narrative. Check if:
1. Backstory events can fit into the narrative timeline
2. Backstory creates prerequisites that aren't met in the narrative
3. Backstory contradicts established facts about earlier events

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Key timeline conflicts or confirmations]"""
            },
            
            # Character psychology
            {
                'type': 'psychology',
                'content': f"""You are a psychologist analyzing character consistency.

CHARACTER BACKSTORY:
{backstory_short}

EVIDENCE FROM NARRATIVE:
{evidence_text}

Analyze whether the character's actions, beliefs, and development in the narrative are consistent with the formative experiences described in the backstory. Consider:
1. Would this backstory produce this personality?
2. Are the character's motivations aligned?
3. Do their decisions make sense given their history?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Key psychological consistencies or conflicts]"""
            },
            
            # Causal chain validation
            {
                'type': 'causal',
                'content': f"""You are analyzing causal relationships between backstory and narrative.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

For each major claim in the backstory, trace its causal implications:
1. What must be true in the narrative if the backstory is true?
2. What cannot be true in the narrative if the backstory is true?
3. Do you find the expected implications or contradictions?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Key causal chains that support or contradict]"""
            },
            
            # Socratic questioning
            {
                'type': 'socratic',
                'content': f"""Use Socratic method to question the backstory's consistency.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

Ask critical questions:
1. What does the backstory claim about the character?
2. What evidence supports or refutes each claim?
3. If the backstory is true, what else must be true?
4. Does the narrative confirm or deny those implications?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Key questions and their answers]"""
            },
            
            # Devil's advocate (assume inconsistent)
            {
                'type': 'devils_advocate_inconsistent',
                'content': f"""Play devil's advocate: Assume the backstory is INCONSISTENT and find evidence to support this.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

Make the strongest possible case that the backstory does NOT fit the narrative.
Then evaluate: Is your case convincing?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Your best arguments and final assessment]"""
            },
            
            # Devil's advocate (assume consistent)
            {
                'type': 'devils_advocate_consistent',
                'content': f"""Play devil's advocate: Assume the backstory IS CONSISTENT and find evidence to support this.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

Make the strongest possible case that the backstory DOES fit the narrative.
Then evaluate: Is your case convincing?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Your best arguments and final assessment]"""
            },
            
            # Step-by-step logical
            {
                'type': 'step_by_step',
                'content': f"""Analyze step-by-step with formal logic.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

Step 1: Extract atomic claims from backstory (C1, C2, C3...)
Step 2: For each claim, find supporting or contradicting evidence
Step 3: Evaluate: Are all claims supported or at least not contradicted?
Step 4: Make final decision

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Your step-by-step analysis]"""
            },
            
            # Counterfactual
            {
                'type': 'counterfactual',
                'content': f"""Use counterfactual reasoning to test consistency.

BACKSTORY:
{backstory_short}

NARRATIVE EVIDENCE:
{evidence_text}

Consider two scenarios:
Scenario A: Backstory is TRUE - what would we expect in the narrative?
Scenario B: Backstory is FALSE - what would be different?

Which scenario better matches the actual narrative evidence?

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Comparison of scenarios]"""
            }
        ]
        
        return prompts
    
    def _format_evidence(self, evidence: List[Dict[str, Any]], max_passages: Optional[int] = None) -> str:
        """Format evidence passages for prompts."""
        if max_passages is None:
            max_passages = len(evidence)

        formatted = []
        for i, ev in enumerate(evidence[:max_passages], 1):
            text = ev.get('text', '')
            formatted.append(f"[Passage {i}]\n{text}\n")
        
        return "\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract decision, confidence, and reasoning."""
        lines = response.strip().split('\n')
        
        decision = None
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            line_upper = line.upper()

            if line_upper.startswith('DECISION:'):
                decision_text = line.split(':', 1)[1].strip().lower()
                if any(t in decision_text for t in ['inconsistent', 'inconistent', 'inconsistant', 'contradict']):
                    decision = 0
                elif 'consistent' in decision_text:
                    decision = 1

            elif line_upper.startswith('CONFIDENCE:'):
                try:
                    conf_text = line.split(':', 1)[1].strip()
                    m = re.search(r"([0-9]*\.?[0-9]+)", conf_text)
                    if m:
                        confidence = float(m.group(1))
                        confidence = max(0.0, min(1.0, confidence))
                except Exception:
                    confidence = 0.5

            elif line_upper.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # If decision wasn't found, try to infer from reasoning
        if decision is None:
            response_lower = response.lower()
            if any(t in response_lower for t in ['inconsistent', 'inconistent', 'inconsistant', 'contradict']):
                decision = 0
            elif 'consistent' in response_lower:
                decision = 1
            else:
                decision = 1  # Default to consistent if unclear
        
        # Collect all reasoning text
        if not reasoning:
            # Strip DECISION/CONFIDENCE lines if the model didn't follow the format
            filtered = []
            for line in lines:
                u = line.strip().upper()
                if u.startswith('DECISION:') or u.startswith('CONFIDENCE:'):
                    continue
                if u.startswith('REASONING:'):
                    filtered.append(line.split(':', 1)[1].strip())
                else:
                    filtered.append(line.strip())
            reasoning = ' '.join([x for x in filtered if x]).strip()
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning if reasoning else response[:200]
        }
    
    def aggregate_chains(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple reasoning chains into final decision.
        
        Args:
            chains: List of reasoning chains
        
        Returns:
            Final decision with aggregated confidence and reasoning
        """
        # Filter out failed chains
        valid_chains = [c for c in chains if c['decision'] is not None]
        
        if not valid_chains:
            logger.error("No valid reasoning chains!")
            return {
                'decision': 1,
                'confidence': 0.0,
                'reasoning': "All reasoning chains failed",
                'chain_votes': {}
            }
        
        if self.voting_strategy == "majority":
            return self._majority_vote(valid_chains)
        elif self.voting_strategy == "weighted":
            return self._weighted_vote(valid_chains)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _majority_vote(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple majority voting."""
        decisions = [c['decision'] for c in chains]
        vote_counts = Counter(decisions)
        
        final_decision = vote_counts.most_common(1)[0][0]
        vote_confidence = vote_counts[final_decision] / len(decisions)
        
        # Get reasoning from chains that agree with majority
        agreeing_chains = [c for c in chains if c['decision'] == final_decision]
        best_reasoning = max(agreeing_chains, key=lambda x: x['confidence'])['reasoning']
        
        logger.info(f"Majority vote: {final_decision} ({vote_confidence:.2%} agreement)")
        
        return {
            'decision': final_decision,
            'confidence': vote_confidence,
            'reasoning': best_reasoning,
            'chain_votes': vote_counts,
            'num_chains': len(chains)
        }
    
    def _weighted_vote(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted voting based on confidence scores."""
        # Calculate weighted votes
        consistent_weight = sum(c['confidence'] for c in chains if c['decision'] == 1)
        inconsistent_weight = sum(c['confidence'] for c in chains if c['decision'] == 0)
        
        total_weight = consistent_weight + inconsistent_weight
        
        if total_weight == 0:
            final_decision = 1
            confidence = 0.5
        else:
            if consistent_weight > inconsistent_weight:
                final_decision = 1
                confidence = consistent_weight / total_weight
            else:
                final_decision = 0
                confidence = inconsistent_weight / total_weight
        
        # Get reasoning from highest-confidence chain
        best_chain = max(chains, key=lambda x: x['confidence'])
        
        logger.info(f"Weighted vote: {final_decision} (confidence: {confidence:.2%})")
        
        return {
            'decision': final_decision,
            'confidence': confidence,
            'reasoning': best_chain['reasoning'],
            'consistent_weight': consistent_weight,
            'inconsistent_weight': inconsistent_weight,
            'num_chains': len(chains)
        }
