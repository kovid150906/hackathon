"""
Self-consistency reasoning engine with multiple independent chains.
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
from loguru import logger
import json
from src.llm_providers import LLMProvider


class SelfConsistencyEngine:
    """Self-consistency reasoning with multiple independent chains."""
    
    def __init__(self, llm_provider: LLMProvider, num_chains: int = 10, voting_strategy: str = "weighted"):
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
                response = self.llm.generate(prompt_text, temperature=0.2)
                parsed = self._parse_response(response)
                chains.append({
                    'chain_id': i,
                    'prompt_type': prompt.get('type', 'unknown') if isinstance(prompt, dict) else 'unknown',
                    'decision': parsed['decision'],
                    'confidence': parsed['confidence'],
                    'reasoning': parsed['reasoning'],
                    'raw_response': response
                })
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
        evidence_text = self._format_evidence(evidence)
        
        prompts = [
            # Direct analysis
            {
                'type': 'direct',
                'content': f"""You are analyzing whether a character's backstory is consistent with a narrative.

BACKSTORY:
{backstory}

EVIDENCE FROM NARRATIVE:
{evidence_text}

Analyze whether the backstory is logically consistent with the narrative. Consider:
1. Direct contradictions (events, facts, character attributes)
2. Causal inconsistencies (backstory prevents later events)
3. Character psychology mismatches
4. Timeline conflicts

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of key evidence supporting your decision]"""
            },
            
            # Timeline reconstruction
            {
                'type': 'timeline',
                'content': f"""You are a detective reconstructing a timeline to verify a character's backstory.

PROPOSED BACKSTORY:
{backstory}

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
{backstory}

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
{backstory}

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
            
            # Contradiction search
            {
                'type': 'contradiction',
                'content': f"""You are specifically searching for contradictions between backstory and narrative.

BACKSTORY CLAIMS:
{backstory}

NARRATIVE EVIDENCE:
{evidence_text}

Your task: Find ANY contradictions between backstory claims and narrative facts. Look for:
1. Explicit factual conflicts
2. Impossible implications
3. Character attribute mismatches
4. World-building inconsistencies

If you find even one clear contradiction, the backstory is INCONSISTENT.

Respond in this format:
DECISION: [CONSISTENT or INCONSISTENT]
CONFIDENCE: [0.0 to 1.0]
REASONING: [List contradictions found, or confirm none found]"""
            },
            
            # Socratic questioning
            {
                'type': 'socratic',
                'content': f"""Use Socratic method to question the backstory's consistency.

BACKSTORY:
{backstory}

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
{backstory}

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
{backstory}

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
{backstory}

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
{backstory}

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
    
    def _format_evidence(self, evidence: List[Dict[str, Any]], max_passages: int = 15) -> str:
        """Format evidence passages for prompts."""
        formatted = []
        for i, ev in enumerate(evidence[:max_passages], 1):
            text = ev.get('text', '')
            score = ev.get('rerank_score', ev.get('score', 0.0))
            formatted.append(f"[Passage {i}] (relevance: {score:.3f})\n{text}\n")
        
        return "\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract decision, confidence, and reasoning."""
        lines = response.strip().split('\n')
        
        decision = None
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('DECISION:'):
                decision_text = line.replace('DECISION:', '').strip().upper()
                if 'CONSISTENT' in decision_text and 'INCONSISTENT' not in decision_text:
                    decision = 1
                elif 'INCONSISTENT' in decision_text:
                    decision = 0
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_text = line.replace('CONFIDENCE:', '').strip()
                    confidence = float(conf_text)
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
            
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        # If decision wasn't found, try to infer from reasoning
        if decision is None:
            response_lower = response.lower()
            if 'inconsistent' in response_lower and 'consistent' not in response_lower:
                decision = 0
            elif 'consistent' in response_lower:
                decision = 1
            else:
                decision = 1  # Default to consistent if unclear
        
        # Collect all reasoning text
        if not reasoning:
            reasoning_started = False
            reasoning_lines = []
            for line in lines:
                if line.startswith('REASONING:'):
                    reasoning_started = True
                    reasoning_lines.append(line.replace('REASONING:', '').strip())
                elif reasoning_started:
                    reasoning_lines.append(line)
            reasoning = ' '.join(reasoning_lines).strip()
        
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
