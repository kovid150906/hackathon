"""
Command-line interface for the Narrative Consistency Checker.
"""
import argparse
import sys
from pathlib import Path
from loguru import logger
from src.pipeline import NarrativeConsistencyChecker


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    logger.add(
        "logs/narrator_{time}.log",
        rotation="100 MB",
        retention="10 days",
        level="DEBUG"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Narrative Consistency Checker - Evaluate backstory consistency with narratives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire dataset
  python main.py --dataset data/ --output results.csv
  
  # Process single example
  python main.py --narrative story.txt --backstory backstory.txt
  
  # Use specific configuration
  python main.py --config config.yaml --dataset data/
  
  # Use Claude instead of Groq
  python main.py --provider anthropic --dataset data/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset directory or file'
    )
    input_group.add_argument(
        '--narrative',
        type=str,
        help='Path to single narrative file (requires --backstory)'
    )
    
    parser.add_argument(
        '--backstory',
        type=str,
        help='Path to backstory file (for single example mode)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['groq', 'ollama', 'anthropic', 'openai', 'google'],
        help='Override LLM provider from config'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Path to output CSV file (default: results.csv)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-self-consistency',
        action='store_true',
        help='Disable self-consistency reasoning'
    )
    
    parser.add_argument(
        '--no-multi-agent',
        action='store_true',
        help='Disable multi-agent system'
    )
    
    parser.add_argument(
        '--no-reranker',
        action='store_true',
        help='Disable evidence reranking'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("=" * 80)
    logger.info("Narrative Consistency Checker - Starting")
    logger.info("=" * 80)
    
    # Validate arguments
    if args.narrative and not args.backstory:
        parser.error("--narrative requires --backstory")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        
        # Override provider if specified
        if args.provider:
            import yaml
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data['llm_provider'] = args.provider
            
            # Save temporary config
            temp_config = Path(args.config).parent / f"temp_config_{args.provider}.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config_data, f)
            
            config_path = str(temp_config)
            logger.info(f"Using provider: {args.provider}")
        else:
            config_path = args.config
        
        # Override feature flags if specified
        if args.no_self_consistency or args.no_multi_agent or args.no_reranker:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if args.no_self_consistency:
                config_data['self_consistency']['enabled'] = False
                logger.info("Self-consistency disabled")
            
            if args.no_multi_agent:
                config_data['multi_agent']['enabled'] = False
                logger.info("Multi-agent system disabled")
            
            if args.no_reranker:
                config_data['reranker']['enabled'] = False
                logger.info("Reranker disabled")
            
            # Save temporary config
            temp_config = Path(config_path).parent / "temp_config_modified.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config_data, f)
            config_path = str(temp_config)
        
        # Initialize checker
        logger.info("Initializing Narrative Consistency Checker")
        checker = NarrativeConsistencyChecker(config_path)
        
        # Process based on mode
        if args.dataset:
            # Dataset mode
            logger.info(f"Processing dataset from {args.dataset}")
            results = checker.process_dataset(args.dataset, args.output)
            
            logger.info("=" * 80)
            logger.info(f"Processing complete! Results saved to {args.output}")
            logger.info(f"Total examples: {len(results)}")
            logger.info(f"Consistent: {(results['Prediction'] == 1).sum()}")
            logger.info(f"Inconsistent: {(results['Prediction'] == 0).sum()}")
            logger.info("=" * 80)
            
        else:
            # Single example mode
            logger.info(f"Processing single example")
            logger.info(f"Narrative: {args.narrative}")
            logger.info(f"Backstory: {args.backstory}")
            
            with open(args.narrative, 'r', encoding='utf-8') as f:
                narrative = f.read()
            
            with open(args.backstory, 'r', encoding='utf-8') as f:
                backstory = f.read()
            
            result = checker.process_single_example(
                narrative_text=narrative,
                backstory=backstory,
                narrative_id=Path(args.narrative).stem
            )
            
            logger.info("=" * 80)
            logger.info("RESULT")
            logger.info("=" * 80)
            logger.info(f"Decision: {'CONSISTENT' if result['decision'] == 1 else 'INCONSISTENT'}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Reasoning: {result['reasoning']}")
            logger.info("=" * 80)
            
            # Save single result
            import pandas as pd
            df = pd.DataFrame([{
                'Story ID': result['narrative_id'],
                'Prediction': result['decision'],
                'Rationale': result['reasoning']
            }])
            df.to_csv(args.output, index=False)
            logger.info(f"Result saved to {args.output}")
        
        logger.info("All done! 🎉")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
