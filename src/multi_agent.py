"""
Multi-agent adversarial reasoning system for narrative consistency checking.
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from src.llm_providers import LLMProvider


class Agent:
    """Base class for reasoning agents."""
    
    def __init__(self, name: str, role: str, llm: LLMProvider, temperature: float = 0.1):
        """Initialize agent."""
        self.name = name
        self.role = role
        self.llm = llm
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []
    
    def analyze(self, backstory: str, evidence: List[Dict[str, Any]], context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze backstory consistency."""
        raise NotImplementedError


class ProsecutorAgent(Agent):
    """Agent that searches for inconsistencies."""
    
    def __init__(self, llm: LLMProvider, temperature: float = 0.2):
        super().__init__(
            name="Prosecutor",
            role="Find evidence AGAINST backstory consistency",
            llm=llm,
            temperature=temperature
        )
    
    def analyze(self, backstory: str, evidence: List[Dict[str, Any]], context: Optional[str] = None) -> Dict[str, Any]:
        """Search for contradictions and inconsistencies."""
        evidence_text = self._format_evidence(evidence)
        
        system_prompt = f"""You are a prosecutor whose job is to find evidence that a character's backstory is INCONSISTENT with the narrative.

Be skeptical and critical. Look for:
1. Direct contradictions (facts, events, character attributes)
2. Impossible implications (backstory makes later events impossible)
3. Character psychology conflicts (backstory incompatible with character's actions)
4. Timeline impossibilities

Your goal is to build the strongest case for INCONSISTENCY."""

        user_prompt = f"""BACKSTORY TO CHALLENGE:
{backstory}

EVIDENCE FROM NARRATIVE:
{evidence_text}

{f"CONTEXT FROM OTHER AGENTS: {context}" if context else ""}

Build your case: What evidence shows this backstory is INCONSISTENT with the narrative?

Provide:
1. ARGUMENT: Your strongest arguments for inconsistency
2. KEY_EVIDENCE: Specific passages that support your case
3. STRENGTH: Rate your case from 0.0 (weak) to 1.0 (very strong)
4. CONCLUSION: CONSISTENT or INCONSISTENT"""

        response = self.llm.generate_with_system(system_prompt, user_prompt, temperature=self.temperature)
        
        parsed = self._parse_response(response)
        parsed['agent'] = self.name
        
        self.history.append({
            'backstory': backstory[:200] + "...",
            'response': response,
            'conclusion': parsed['conclusion']
        })
        
        logger.info(f"Prosecutor: {parsed['conclusion']} (strength: {parsed['strength']:.2f})")
        return parsed
    
    def _format_evidence(self, evidence: List[Dict[str, Any]], max_passages: int = 10) -> str:
        """Format evidence passages."""
        formatted = []
        for i, ev in enumerate(evidence[:max_passages], 1):
            text = ev.get('text', '')
            score = ev.get('rerank_score', ev.get('score', 0.0))
            formatted.append(f"[Evidence {i}] (relevance: {score:.3f})\n{text}\n")
        return "\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response."""
        lines = response.strip().split('\n')
        
        argument = ""
        key_evidence = ""
        strength = 0.5
        conclusion = None
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('ARGUMENT:') or line_stripped.startswith('1. ARGUMENT:'):
                current_section = 'argument'
                argument = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif line_stripped.startswith('KEY_EVIDENCE:') or line_stripped.startswith('2. KEY_EVIDENCE:'):
                current_section = 'evidence'
                key_evidence = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif line_stripped.startswith('STRENGTH:') or line_stripped.startswith('3. STRENGTH:'):
                current_section = 'strength'
                try:
                    strength_text = line_stripped.split(':', 1)[1].strip()
                    strength = float(strength_text.split()[0])
                    strength = max(0.0, min(1.0, strength))
                except:
                    strength = 0.5
            elif line_stripped.startswith('CONCLUSION:') or line_stripped.startswith('4. CONCLUSION:'):
                current_section = 'conclusion'
                conclusion_text = line_stripped.split(':', 1)[1].strip().upper()
                if 'INCONSISTENT' in conclusion_text:
                    conclusion = 0
                elif 'CONSISTENT' in conclusion_text:
                    conclusion = 1
            elif current_section and line_stripped:
                if current_section == 'argument':
                    argument += " " + line_stripped
                elif current_section == 'evidence':
                    key_evidence += " " + line_stripped
        
        if conclusion is None:
            conclusion = 0 if strength > 0.6 else 1
        
        return {
            'argument': argument or response[:300],
            'key_evidence': key_evidence,
            'strength': strength,
            'conclusion': conclusion,
            'raw_response': response
        }


class DefenderAgent(Agent):
    """Agent that searches for supporting evidence."""
    
    def __init__(self, llm: LLMProvider, temperature: float = 0.2):
        super().__init__(
            name="Defender",
            role="Find evidence FOR backstory consistency",
            llm=llm,
            temperature=temperature
        )
    
    def analyze(self, backstory: str, evidence: List[Dict[str, Any]], context: Optional[str] = None) -> Dict[str, Any]:
        """Search for supporting evidence and consistency."""
        evidence_text = self._format_evidence(evidence)
        
        system_prompt = f"""You are a defender whose job is to find evidence that a character's backstory IS CONSISTENT with the narrative.

Be thorough and supportive. Look for:
1. Supporting evidence (facts that confirm backstory claims)
2. Absence of contradictions
3. Character psychology alignment
4. Causal chains that make sense

Your goal is to build the strongest case for CONSISTENCY."""

        user_prompt = f"""BACKSTORY TO DEFEND:
{backstory}

EVIDENCE FROM NARRATIVE:
{evidence_text}

{f"CONTEXT FROM OTHER AGENTS: {context}" if context else ""}

Build your case: What evidence shows this backstory IS CONSISTENT with the narrative?

Provide:
1. ARGUMENT: Your strongest arguments for consistency
2. KEY_EVIDENCE: Specific passages that support your case
3. STRENGTH: Rate your case from 0.0 (weak) to 1.0 (very strong)
4. CONCLUSION: CONSISTENT or INCONSISTENT"""

        response = self.llm.generate_with_system(system_prompt, user_prompt, temperature=self.temperature)
        
        parsed = self._parse_response(response)
        parsed['agent'] = self.name
        
        self.history.append({
            'backstory': backstory[:200] + "...",
            'response': response,
            'conclusion': parsed['conclusion']
        })
        
        logger.info(f"Defender: {parsed['conclusion']} (strength: {parsed['strength']:.2f})")
        return parsed
    
    def _format_evidence(self, evidence: List[Dict[str, Any]], max_passages: int = 10) -> str:
        """Format evidence passages."""
        formatted = []
        for i, ev in enumerate(evidence[:max_passages], 1):
            text = ev.get('text', '')
            score = ev.get('rerank_score', ev.get('score', 0.0))
            formatted.append(f"[Evidence {i}] (relevance: {score:.3f})\n{text}\n")
        return "\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response."""
        lines = response.strip().split('\n')
        
        argument = ""
        key_evidence = ""
        strength = 0.5
        conclusion = None
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('ARGUMENT:') or line_stripped.startswith('1. ARGUMENT:'):
                current_section = 'argument'
                argument = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif line_stripped.startswith('KEY_EVIDENCE:') or line_stripped.startswith('2. KEY_EVIDENCE:'):
                current_section = 'evidence'
                key_evidence = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif line_stripped.startswith('STRENGTH:') or line_stripped.startswith('3. STRENGTH:'):
                current_section = 'strength'
                try:
                    strength_text = line_stripped.split(':', 1)[1].strip()
                    strength = float(strength_text.split()[0])
                    strength = max(0.0, min(1.0, strength))
                except:
                    strength = 0.5
            elif line_stripped.startswith('CONCLUSION:') or line_stripped.startswith('4. CONCLUSION:'):
                current_section = 'conclusion'
                conclusion_text = line_stripped.split(':', 1)[1].strip().upper()
                if 'CONSISTENT' in conclusion_text and 'INCONSISTENT' not in conclusion_text:
                    conclusion = 1
                elif 'INCONSISTENT' in conclusion_text:
                    conclusion = 0
            elif current_section and line_stripped:
                if current_section == 'argument':
                    argument += " " + line_stripped
                elif current_section == 'evidence':
                    key_evidence += " " + line_stripped
        
        if conclusion is None:
            conclusion = 1 if strength > 0.6 else 0
        
        return {
            'argument': argument or response[:300],
            'key_evidence': key_evidence,
            'strength': strength,
            'conclusion': conclusion,
            'raw_response': response
        }


class InvestigatorAgent(Agent):
    """Neutral agent that gathers facts."""
    
    def __init__(self, llm: LLMProvider, temperature: float = 0.0):
        super().__init__(
            name="Investigator",
            role="Neutral fact-finding and evidence gathering",
            llm=llm,
            temperature=temperature
        )
    
    def analyze(self, backstory: str, evidence: List[Dict[str, Any]], context: Optional[str] = None) -> Dict[str, Any]:
        """Gather and organize facts neutrally."""
        evidence_text = self._format_evidence(evidence)
        
        system_prompt = """You are a neutral investigator who gathers facts without bias.

Your job is to:
1. Extract key claims from the backstory
2. Find relevant evidence for each claim
3. Report facts objectively without judgment
4. Identify gaps in evidence"""

        user_prompt = f"""BACKSTORY:
{backstory}

NARRATIVE EVIDENCE:
{evidence_text}

{f"CONTEXT: {context}" if context else ""}

Conduct a neutral investigation:
1. CLAIMS: List key claims from backstory
2. EVIDENCE: For each claim, note supporting or contradicting evidence
3. GAPS: What questions remain unanswered?
4. ASSESSMENT: Based on facts only, does evidence support or contradict the backstory?"""

        response = self.llm.generate_with_system(system_prompt, user_prompt, temperature=self.temperature)
        
        parsed = self._parse_investigator_response(response)
        parsed['agent'] = self.name
        
        logger.info(f"Investigator: {parsed['assessment']}")
        return parsed
    
    def _format_evidence(self, evidence: List[Dict[str, Any]], max_passages: int = 15) -> str:
        """Format evidence passages."""
        formatted = []
        for i, ev in enumerate(evidence[:max_passages], 1):
            text = ev.get('text', '')
            score = ev.get('rerank_score', ev.get('score', 0.0))
            formatted.append(f"[Evidence {i}] (relevance: {score:.3f})\n{text}\n")
        return "\n".join(formatted)
    
    def _parse_investigator_response(self, response: str) -> Dict[str, Any]:
        """Parse investigator response."""
        return {
            'claims': self._extract_section(response, 'CLAIMS'),
            'evidence_analysis': self._extract_section(response, 'EVIDENCE'),
            'gaps': self._extract_section(response, 'GAPS'),
            'assessment': self._extract_section(response, 'ASSESSMENT'),
            'raw_response': response
        }
    
    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a section from response."""
        lines = response.split('\n')
        section_text = []
        in_section = False
        
        for line in lines:
            if section_name in line.upper():
                in_section = True
                section_text.append(line.split(':', 1)[1].strip() if ':' in line else "")
            elif in_section:
                if any(s in line.upper() for s in ['CLAIMS:', 'EVIDENCE:', 'GAPS:', 'ASSESSMENT:']) and line.strip().endswith(':'):
                    break
                section_text.append(line.strip())
        
        return ' '.join(section_text).strip()


class JudgeAgent(Agent):
    """Judge agent that makes final decision."""
    
    def __init__(self, llm: LLMProvider, temperature: float = 0.0):
        super().__init__(
            name="Judge",
            role="Final decision based on all evidence and arguments",
            llm=llm,
            temperature=temperature
        )
    
    def make_decision(self, backstory: str, prosecutor_case: Dict[str, Any], 
                      defender_case: Dict[str, Any], investigator_report: Dict[str, Any]) -> Dict[str, Any]:
        """Make final decision based on all agent inputs."""
        
        system_prompt = """You are an impartial judge making a final decision on backstory consistency.

You have heard arguments from:
1. The Prosecutor (arguing for INCONSISTENCY)
2. The Defender (arguing for CONSISTENCY)
3. The Investigator (neutral fact-finding)

Your job is to weigh all evidence and make a final, definitive ruling."""

        user_prompt = f"""BACKSTORY UNDER REVIEW:
{backstory}

PROSECUTOR'S CASE (against consistency):
{prosecutor_case['argument']}
Strength: {prosecutor_case['strength']:.2f}
Conclusion: {'INCONSISTENT' if prosecutor_case['conclusion'] == 0 else 'CONSISTENT'}

DEFENDER'S CASE (for consistency):
{defender_case['argument']}
Strength: {defender_case['strength']:.2f}
Conclusion: {'CONSISTENT' if defender_case['conclusion'] == 1 else 'INCONSISTENT'}

INVESTIGATOR'S REPORT:
{investigator_report['assessment']}

Now make your ruling:
1. VERDICT: CONSISTENT or INCONSISTENT
2. CONFIDENCE: 0.0 to 1.0
3. REASONING: Explain which arguments were most persuasive and why
4. KEY_EVIDENCE: The most critical evidence that decided your verdict"""

        response = self.llm.generate_with_system(system_prompt, user_prompt, temperature=self.temperature)
        
        parsed = self._parse_response(response)
        parsed['agent'] = self.name
        parsed['prosecutor_strength'] = prosecutor_case['strength']
        parsed['defender_strength'] = defender_case['strength']
        
        logger.info(f"Judge: {parsed['verdict']} (confidence: {parsed['confidence']:.2f})")
        return parsed
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse judge response."""
        lines = response.strip().split('\n')
        
        verdict = None
        confidence = 0.5
        reasoning = ""
        key_evidence = ""
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('VERDICT:') or line_stripped.startswith('1. VERDICT:'):
                current_section = 'verdict'
                verdict_text = line_stripped.split(':', 1)[1].strip().upper()
                if 'INCONSISTENT' in verdict_text:
                    verdict = 0
                elif 'CONSISTENT' in verdict_text:
                    verdict = 1
            elif line_stripped.startswith('CONFIDENCE:') or line_stripped.startswith('2. CONFIDENCE:'):
                current_section = 'confidence'
                try:
                    conf_text = line_stripped.split(':', 1)[1].strip()
                    confidence = float(conf_text.split()[0])
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5
            elif line_stripped.startswith('REASONING:') or line_stripped.startswith('3. REASONING:'):
                current_section = 'reasoning'
                reasoning = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif line_stripped.startswith('KEY_EVIDENCE:') or line_stripped.startswith('4. KEY_EVIDENCE:'):
                current_section = 'evidence'
                key_evidence = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
            elif current_section and line_stripped:
                if current_section == 'reasoning':
                    reasoning += " " + line_stripped
                elif current_section == 'evidence':
                    key_evidence += " " + line_stripped
        
        if verdict is None:
            verdict = 1 if confidence > 0.5 else 0
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning or response[:300],
            'key_evidence': key_evidence,
            'raw_response': response
        }


class MultiAgentSystem:
    """Multi-agent adversarial reasoning system."""
    
    def __init__(self, llm: LLMProvider, config: Dict[str, Any]):
        """Initialize multi-agent system."""
        self.llm = llm
        self.config = config
        
        agent_configs = config.get('agents', {})
        
        self.prosecutor = ProsecutorAgent(
            llm,
            temperature=agent_configs.get('prosecutor', {}).get('temperature', 0.2)
        )
        self.defender = DefenderAgent(
            llm,
            temperature=agent_configs.get('defender', {}).get('temperature', 0.2)
        )
        self.investigator = InvestigatorAgent(
            llm,
            temperature=agent_configs.get('investigator', {}).get('temperature', 0.0)
        )
        self.judge = JudgeAgent(
            llm,
            temperature=agent_configs.get('judge', {}).get('temperature', 0.0)
        )
        
        self.deliberation_rounds = config.get('deliberation_rounds', 3)
        
        logger.info("Initialized MultiAgentSystem with 4 agents")
    
    def deliberate(self, backstory: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multi-agent deliberation with parallel initial analyses."""
        logger.info(f"Starting multi-agent deliberation")
        
        # Run initial analyses in parallel for speed
        with ThreadPoolExecutor(max_workers=3) as executor:
            prosecutor_future = executor.submit(self.prosecutor.analyze, backstory, evidence)
            defender_future = executor.submit(self.defender.analyze, backstory, evidence)
            investigator_future = executor.submit(self.investigator.analyze, backstory, evidence)
            
            prosecutor_case = prosecutor_future.result()
            defender_case = defender_future.result()
            investigator_report = investigator_future.result()
        
        logger.info("✓ All agents completed initial analysis")
        
        # Judge makes final decision
        final_decision = self.judge.make_decision(
            backstory,
            prosecutor_case,
            defender_case,
            investigator_report
        )
        
        return {
            'decision': final_decision['verdict'],
            'confidence': final_decision['confidence'],
            'reasoning': final_decision['reasoning'],
            'prosecutor_case': prosecutor_case,
            'defender_case': defender_case,
            'investigator_report': investigator_report,
            'judge_decision': final_decision
        }
