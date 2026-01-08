"""
Ensemble voting and decision logic for combining multiple models and methods.
"""
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
from loguru import logger
from src.llm_providers import create_llm_provider


class EnsembleVoter:
    """Ensemble voting system for combining multiple predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble voter."""
        self.config = config
        self.voting_strategy = config.get('voting_strategy', 'weighted')
        self.min_agreement = config.get('min_agreement', 0.6)
        
        logger.info(f"Initialized EnsembleVoter with strategy: {self.voting_strategy}")
    
    def vote(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple predictions into final decision.
        
        Args:
            predictions: List of predictions from different models/methods
                Each prediction should have: decision, confidence, reasoning, weight (optional)
        
        Returns:
            Final aggregated decision
        """
        if not predictions:
            logger.error("No predictions to vote on!")
            return {
                'decision': 1,
                'confidence': 0.0,
                'reasoning': "No predictions available",
                'method': 'none'
            }
        
        if len(predictions) == 1:
            result = predictions[0].copy()
            result['method'] = 'single'
            return result
        
        if self.voting_strategy == 'majority':
            return self._majority_vote(predictions)
        elif self.voting_strategy == 'weighted':
            return self._weighted_vote(predictions)
        elif self.voting_strategy == 'soft':
            return self._soft_vote(predictions)
        else:
            logger.warning(f"Unknown voting strategy: {self.voting_strategy}, using weighted")
            return self._weighted_vote(predictions)
    
    def _majority_vote(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple majority voting (each prediction counts equally)."""
        decisions = [p['decision'] for p in predictions]
        vote_counts = Counter(decisions)
        
        # Get majority decision
        final_decision = vote_counts.most_common(1)[0][0]
        agreement_rate = vote_counts[final_decision] / len(predictions)
        
        # Average confidence of agreeing predictions
        agreeing_preds = [p for p in predictions if p['decision'] == final_decision]
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in agreeing_preds])
        
        # Get best reasoning from agreeing predictions
        best_reasoning = max(agreeing_preds, key=lambda x: x.get('confidence', 0.5))['reasoning']
        
        logger.info(f"Majority vote: {final_decision} ({agreement_rate:.2%} agreement, confidence: {avg_confidence:.2f})")
        
        return {
            'decision': final_decision,
            'confidence': avg_confidence * agreement_rate,  # Adjusted by agreement
            'reasoning': best_reasoning,
            'agreement_rate': agreement_rate,
            'vote_counts': dict(vote_counts),
            'method': 'majority'
        }
    
    def _weighted_vote(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted voting based on confidence and model weights."""
        # Calculate weighted votes for each decision
        decision_weights = {0: 0.0, 1: 0.0}
        
        for pred in predictions:
            decision = pred['decision']
            confidence = pred.get('confidence', 0.5)
            model_weight = pred.get('weight', 1.0)
            
            # Combined weight = confidence * model_weight
            combined_weight = confidence * model_weight
            decision_weights[decision] += combined_weight
        
        # Determine winner
        if decision_weights[1] > decision_weights[0]:
            final_decision = 1
            final_confidence = decision_weights[1] / (decision_weights[0] + decision_weights[1])
        else:
            final_decision = 0
            final_confidence = decision_weights[0] / (decision_weights[0] + decision_weights[1])
        
        # Get reasoning from highest-weight agreeing prediction
        agreeing_preds = [p for p in predictions if p['decision'] == final_decision]
        best_pred = max(agreeing_preds, key=lambda x: x.get('confidence', 0.5) * x.get('weight', 1.0))
        
        logger.info(f"Weighted vote: {final_decision} (confidence: {final_confidence:.2f}, weights: {decision_weights})")
        
        return {
            'decision': final_decision,
            'confidence': final_confidence,
            'reasoning': best_pred['reasoning'],
            'decision_weights': decision_weights,
            'num_predictions': len(predictions),
            'method': 'weighted'
        }
    
    def _soft_vote(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Soft voting using continuous confidence scores."""
        # Calculate average confidence for each decision class
        class_confidences = {0: [], 1: []}
        
        for pred in predictions:
            decision = pred['decision']
            confidence = pred.get('confidence', 0.5)
            weight = pred.get('weight', 1.0)
            
            # Add weighted confidence
            class_confidences[decision].append(confidence * weight)
        
        # Calculate mean confidence for each class
        mean_confidences = {
            0: np.mean(class_confidences[0]) if class_confidences[0] else 0.0,
            1: np.mean(class_confidences[1]) if class_confidences[1] else 0.0
        }
        
        # Normalize to get probability-like scores
        total = mean_confidences[0] + mean_confidences[1]
        if total > 0:
            prob_inconsistent = mean_confidences[0] / total
            prob_consistent = mean_confidences[1] / total
        else:
            prob_inconsistent = prob_consistent = 0.5
        
        # Final decision
        if prob_consistent > prob_inconsistent:
            final_decision = 1
            final_confidence = prob_consistent
        else:
            final_decision = 0
            final_confidence = prob_inconsistent
        
        # Get reasoning from highest-confidence prediction
        best_pred = max(predictions, key=lambda x: x.get('confidence', 0.5))
        
        logger.info(f"Soft vote: {final_decision} (p(consistent)={prob_consistent:.2f}, p(inconsistent)={prob_inconsistent:.2f})")
        
        return {
            'decision': final_decision,
            'confidence': final_confidence,
            'reasoning': best_pred['reasoning'],
            'class_probabilities': {
                'consistent': prob_consistent,
                'inconsistent': prob_inconsistent
            },
            'method': 'soft'
        }


class MultiModelEnsemble:
    """Ensemble system that runs multiple models in parallel."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-model ensemble."""
        self.config = config
        self.models = []
        
        # Initialize multiple LLM providers
        for model_config in config.get('models', []):
            provider_name = model_config['provider']
            model_name = model_config['model']
            weight = model_config.get('weight', 1.0)
            
            try:
                from src.config import get_config
                full_config = get_config()
                provider_config = full_config.get_provider_config(provider_name)
                provider_config['model'] = model_name
                
                llm = create_llm_provider(provider_name, provider_config)
                
                self.models.append({
                    'provider': provider_name,
                    'model': model_name,
                    'llm': llm,
                    'weight': weight
                })
                
                logger.info(f"Added model to ensemble: {provider_name}/{model_name} (weight: {weight})")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}/{model_name}: {e}")
        
        if not self.models:
            logger.warning("No models initialized in ensemble!")
        
        self.voter = EnsembleVoter(config)
    
    def predict(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get predictions from all models in ensemble.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            List of predictions from each model
        """
        predictions = []
        
        for model_info in self.models:
            try:
                logger.info(f"Getting prediction from {model_info['provider']}/{model_info['model']}")
                
                if system_prompt:
                    response = model_info['llm'].generate_with_system(system_prompt, prompt)
                else:
                    response = model_info['llm'].generate(prompt)
                
                # Parse response (basic parsing - should be customized based on task)
                parsed = self._parse_response(response)
                parsed['provider'] = model_info['provider']
                parsed['model'] = model_info['model']
                parsed['weight'] = model_info['weight']
                
                predictions.append(parsed)
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_info['provider']}: {e}")
        
        return predictions
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Basic response parsing."""
        # This is a simple parser - should be customized based on actual response format
        decision = 1
        confidence = 0.5
        
        response_lower = response.lower()
        
        if 'inconsistent' in response_lower and 'consistent' not in response_lower:
            decision = 0
        elif 'consistent' in response_lower:
            decision = 1
        
        # Try to extract confidence
        for line in response.split('\n'):
            if 'confidence' in line.lower():
                try:
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        confidence = float(match.group(1))
                        if confidence > 1:
                            confidence = confidence / 100
                        confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': response[:300]
        }
    
    def ensemble_predict(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ensemble prediction from all models.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            Aggregated ensemble prediction
        """
        predictions = self.predict(prompt, system_prompt)
        
        if not predictions:
            return {
                'decision': 1,
                'confidence': 0.0,
                'reasoning': "No model predictions available",
                'method': 'none'
            }
        
        # Aggregate using voter
        final_decision = self.voter.vote(predictions)
        final_decision['individual_predictions'] = predictions
        
        return final_decision
