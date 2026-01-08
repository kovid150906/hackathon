# Usage Examples

## Basic Usage

### 1. Process Entire Dataset

```powershell
python main.py --dataset data/ --output results.csv
```

This will:
- Load all narratives and backstories from `data/`
- Process each example through the full pipeline
- Save results to `results.csv`

### 2. Process Single Example

```powershell
python main.py `
    --narrative examples/pride_and_prejudice.txt `
    --backstory examples/elizabeth_backstory.txt `
    --output single_result.csv
```

## Advanced Usage

### Using Different LLM Providers

#### Groq (FREE, Fast)
```powershell
python main.py --provider groq --dataset data/
```

#### Ollama (FREE, Local)
```powershell
# Make sure Ollama is running
ollama serve

# Run with Ollama
python main.py --provider ollama --dataset data/
```

#### Claude 3.5 Sonnet (PAID, Best Accuracy)
```powershell
# Make sure ANTHROPIC_API_KEY is in .env
python main.py --provider anthropic --dataset data/
```

### Performance Tuning

#### Fast Mode (Speed Priority)
```powershell
python main.py `
    --dataset data/ `
    --no-multi-agent `
    --no-self-consistency `
    --output results_fast.csv
```

#### Accuracy Mode (Quality Priority)
```powershell
python main.py `
    --provider anthropic `
    --dataset data/ `
    --output results_accurate.csv
```

### Configuration Options

#### Custom Configuration File
```powershell
python main.py `
    --config custom_config.yaml `
    --dataset data/
```

#### Disable Specific Features
```powershell
# Disable self-consistency
python main.py --no-self-consistency --dataset data/

# Disable multi-agent
python main.py --no-multi-agent --dataset data/

# Disable reranker
python main.py --no-reranker --dataset data/

# Disable all advanced features (fastest)
python main.py `
    --no-self-consistency `
    --no-multi-agent `
    --no-reranker `
    --dataset data/
```

### Logging and Debugging

#### Debug Mode
```powershell
python main.py --log-level DEBUG --dataset data/
```

#### Check Logs
```powershell
# View latest log file
notepad logs\narrator_*.log
```

## Dataset Format Examples

### Option 1: Directory Structure

```
data/
├── story1.txt
├── story1_backstory.txt
├── story2.txt
├── story2_backstory.txt
└── ...
```

### Option 2: JSON Format

```json
[
    {
        "id": "story1",
        "narrative": "Full narrative text here...",
        "backstory": "Character backstory here..."
    },
    {
        "id": "story2",
        "narrative": "Another narrative...",
        "backstory": "Another backstory..."
    }
]
```

Save as `dataset.json` and run:
```powershell
python main.py --dataset dataset.json --output results.csv
```

### Option 3: JSONL Format

```jsonl
{"id": "story1", "narrative": "...", "backstory": "..."}
{"id": "story2", "narrative": "...", "backstory": "..."}
```

Save as `dataset.jsonl` and run:
```powershell
python main.py --dataset dataset.jsonl --output results.csv
```

## Output Examples

### CSV Output Format

```csv
Story ID,Prediction,Rationale
story1,1,"The backstory aligns with the character's development. Key evidence: the formative experience mentioned matches the character's later behavior..."
story2,0,"Timeline conflict detected. The backstory claims the character was in London in 1805, but the narrative establishes they were in Paris..."
```

### Interpreting Results

- **Prediction = 1**: Backstory is CONSISTENT with narrative
- **Prediction = 0**: Backstory is INCONSISTENT with narrative
- **Rationale**: Brief explanation (automatically generated)

## Real-World Workflow

### Full Hackathon Submission Process

```powershell
# 1. Setup (one time)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Download dataset
# (Download from organizers and place in data/)

# 4. Test on single example first
python main.py `
    --narrative data/story1.txt `
    --backstory data/story1_backstory.txt `
    --output test_result.csv

# 5. Run on full dataset
python main.py --dataset data/ --output results.csv

# 6. Check results
notepad results.csv

# 7. (Optional) Try with better model if accuracy is low
python main.py `
    --provider anthropic `
    --dataset data/ `
    --output results_claude.csv

# 8. Package for submission
# Create <TEAMNAME>_KDSH_2026.zip with:
#   - All code
#   - results.csv
#   - report.pdf (10 pages max)
```

## Testing and Validation

### Test Single Components

```powershell
# Test embeddings
python -c "from src.pathway_ingestion import PathwayVectorStore; print('✓ Vector store OK')"

# Test LLM connection
python -c "from src.llm_providers import create_llm_provider; from src.config import get_config; cfg = get_config(); llm = create_llm_provider('groq', cfg.get_provider_config()); print(llm.generate('Hello')); print('✓ LLM OK')"

# Test full pipeline
python main.py --narrative test.txt --backstory test_backstory.txt
```

## Tips for Best Results

### 1. For Maximum Accuracy
- Use all features enabled (default config)
- Use Groq Llama 3.1 70B or Claude 3.5
- Increase `num_chains` in config to 15-20

### 2. For Speed
- Disable multi-agent: `--no-multi-agent`
- Reduce reasoning chains in config: `num_chains: 3`
- Use Groq (fastest free option)

### 3. For Cost Optimization
- Use Groq (completely free)
- Or use Ollama (local, no API costs)
- Only upgrade to Claude for final submission

### 4. For Debugging
- Use `--log-level DEBUG`
- Process single example first
- Check logs in `logs/` directory
