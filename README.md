# Narrative Consistency Checker 🔍

> **Track A Solution** for Kharagpur Data Science Hackathon 2026
>
> A sophisticated multi-agent system for evaluating backstory consistency with long-form narratives using Pathway framework and multiple LLM reasoning strategies.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This system determines whether a hypothetical character backstory is **consistent** or **inconsistent** with a complete narrative (100k+ words) by:

1. **Pathway-based Vector Store**: Semantic chunking and hybrid retrieval
2. **Self-Consistency Reasoning**: Multiple independent reasoning chains with varied prompting strategies
3. **Multi-Agent Adversarial System** (optional): Prosecutor, Defender, Investigator, and Judge agents
4. **Ensemble Voting** (optional): Weighted aggregation of multiple reasoning methods
5. **Flexible LLM Providers**: HuggingFace (FREE - default), Groq (FREE), Ollama (FREE), DeepSeek (FREE), plus Claude, GPT-4, and Gemini
6. **Smart Fallback Chain**: Automatic failover between providers (HuggingFace → Groq → Ollama) for maximum reliability

## 🏗️ Architecture

```
Input: Narrative (100k+ words) + Hypothetical Backstory
    ↓
┌─────────────────────────────────────────────────────┐
│ 1. Pathway Ingestion & Vector Store                 │
│    - Semantic/hybrid chunking                       │
│    - BGE-large embeddings                           │
│    - Hybrid search (semantic + keyword)             │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2. Evidence Retrieval & Reranking                   │
│    - Extract queries from backstory                 │
│    - Multi-query retrieval                          │
│    - Cross-encoder reranking                        │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3. Multi-Method Reasoning                           │
│    ┌─────────────────────────────────────┐          │
│    │ Self-Consistency (2 chains default) │          │
│    │ - Direct analysis                   │          │
│    │ - Timeline reconstruction           │          │
│    │ - Character psychology              │          │
│    │ - Causal validation                 │          │
│    │ - Contradiction search              │          │
│    │ - Socratic questioning              │          │
│    │ - Devil's advocate (both sides)     │          │
│    │ - Step-by-step logic                │          │
│    │ - Counterfactual reasoning          │          │
│    │ - Early stopping when confident     │          │
│    └─────────────────────────────────────┘          │
│    ┌─────────────────────────────────────┐          │
│    │ Multi-Agent System (optional)       │          │
│    │ - Prosecutor: Find inconsistencies  │          │
│    │ - Defender: Find supporting evidence│          │
│    │ - Investigator: Neutral fact-finding│          │
│    │ - Judge: Final decision             │          │
│    └─────────────────────────────────────┘          │
│    ┌─────────────────────────────────────┐          │
│    │ Smart Fallback (HF → Groq → Ollama)│          │
│    │ - Automatic provider switching      │          │
│    │ - Handles rate limits gracefully    │          │
│    └─────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 4. Ensemble Voting                                  │
│    - Weighted aggregation                           │
│    - Confidence calibration                         │
└─────────────────────────────────────────────────────┘
    ↓
Output: Decision (0/1) + Confidence + Reasoning
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- **Windows users**: WSL (Windows Subsystem for Linux) required for Pathway
- **Linux/Mac users**: Native support
- (Optional) CUDA-capable GPU for local embeddings

### Installation

#### **For Windows Users** (Recommended: WSL)

Pathway framework requires Linux. Use WSL (built into Windows 10/11):

**Step 1: Install WSL**

```powershell
# Open PowerShell as Administrator and run:
wsl --install

# This installs Ubuntu by default. Restart your computer if prompted.
```

**Step 2: Verify WSL Installation**

```powershell
# Open a new PowerShell window and check:
wsl --version

# Should show WSL version info. If not, restart your computer.
```

**Step 3: Start WSL and Navigate to Project**

```powershell
# Open WSL Ubuntu terminal:
wsl

# You're now in Linux! Navigate to your project folder:
# Windows drives are mounted under /mnt/
# Example: D:\Learning\Kharapur_Hackathon\hackathon becomes:
cd /mnt/d/Learning/Kharapur_Hackathon/hackathon

# Verify you're in the right folder:
ls -la
# Should see: main.py, config.yaml, requirements.txt, etc.
```

**Step 4: Install Python and System Dependencies**

```bash
# Update package list and install Python:
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Verify Python installation:
python3 --version
# Should show: Python 3.x.x
```

**Step 5: Create Virtual Environment**

```bash
# Create virtual environment:
python3 -m venv venv

# Activate it:
source venv/bin/activate

# Your prompt should now show (venv) prefix
```

**Step 6: Install Python Dependencies**

```bash
# Upgrade pip first:
pip install --upgrade pip

# Install all requirements (this may take 5-10 minutes):
pip install -r requirements.txt

# Verify key packages installed:
pip list | grep -E 'pathway|groq|sentence-transformers'
```

#### **For Linux/Mac Users**

**Step 1: Navigate to Project**

```bash
# If you cloned from git:
cd path/to/hackathon

# Or if you have the folder already:
cd /path/to/Kharapur_Hackathon/hackathon

# Verify you're in the right folder:
ls -la
# Should see: main.py, config.yaml, requirements.txt, etc.
```

**Step 2: Check Python Installation**

```bash
# Verify Python 3.8+ is installed:
python3 --version

# If not installed:
# Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv
# macOS: brew install python3
```

**Step 3: Create Virtual Environment**

```bash
# Create virtual environment:
python3 -m venv venv

# Activate it:
source venv/bin/activate

# Your prompt should now show (venv) prefix
```

**Step 4: Install Dependencies**

```bash
# Upgrade pip:
pip install --upgrade pip

# Install all requirements (5-10 minutes):
pip install -r requirements.txt

# Verify installation:
pip list | grep -E 'pathway|groq|sentence-transformers'
```

### Setup API Keys

**Step 1: Get FREE API Keys** (At least one required, multiple recommended)

1. **HuggingFace** (default, recommended):

   - Go to: https://huggingface.co/settings/tokens
   - Click "New token" → Name it (e.g., "hackathon") → Copy the token
   - Starts with `hf_...`

2. **Groq** (fast, recommended):

   - Go to: https://console.groq.com
   - Sign up/login → Go to API Keys → Create new key
   - Starts with `gsk_...`

3. **DeepSeek** (optional):
   - Go to: https://platform.deepseek.com
   - Sign up → Get API key

**Step 2: Create `.env` File**

```bash
# In your project folder (WSL/Linux/Mac):
cp .env.example .env

# Edit the file:
nano .env
# OR use your favorite editor:
code .env  # VS Code
vim .env   # Vim
```

**Step 3: Add Your API Keys**

Open `.env` and replace the placeholder values:

```bash
# Required: At least one of these
HUGGINGFACE_API_KEY=hf_your_actual_huggingface_token_here
GROQ_API_KEY=gsk_your_actual_groq_key_here

# Optional but recommended:
DEEPSEEK_API_KEY=your_actual_deepseek_key_here

# Only if using local Ollama:
OLLAMA_MODEL=phi3:mini

# Optional paid providers:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...
```

**Step 4: Verify API Keys Work**

```bash
# Test that keys are loaded:
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF Key:', 'SET' if os.getenv('HUGGINGFACE_API_KEY') else 'NOT SET'); print('Groq Key:', 'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET')"

# Should show:
# HF Key: SET
# Groq Key: SET
```

**Important**: The system uses a smart fallback chain (HuggingFace → Groq → Ollama). Having multiple API keys ensures the system keeps running if one provider hits rate limits.

### Run the System

**Important**: Make sure you're in the project directory with virtual environment activated:

```bash
# Should see (venv) in your prompt
# If not, run:
source venv/bin/activate  # Linux/Mac/WSL
```

#### **Quick Test** (Single Example)

Test the system on one example first:

```bash
# Run the quick check script:
python scripts/quick_check.py

# This processes the first row of train.csv
# Expected output (after 10-30 seconds):
# decision= 0 or 1
# confidence= 0.XX
# evidence_count= 20
```

If this works, your setup is complete! ✅

#### **Process Training Dataset** (140 examples)

```bash
# Process all training data:
python main.py --dataset train.csv --output train_results.csv

# Expected behavior:
# - Shows progress bar
# - Takes 30-60 minutes depending on provider
# - Creates train_results.csv with predictions
# - Can resume if interrupted (skips already processed IDs)

# Check the output:
head -5 train_results.csv
# Should show:
# Story ID,Prediction,Rationale
# 46,1,"No contradictions found..."
# 137,0,"Backstory contradicts..."
```

#### **Generate Test Predictions** (For Submission)

```bash
# Process test dataset (59 examples):
python main.py --dataset test.csv --output submission.csv

# This is your final submission file!
# Expected time: 15-30 minutes

# Verify submission format:
head -5 submission.csv
wc -l submission.csv
# Should show 60 lines (1 header + 59 predictions)
```

#### **Resume Interrupted Run**

If the process stops (rate limit, crash, etc.), just rerun the same command:

```bash
# It automatically skips already processed examples:
python main.py --dataset test.csv --output submission.csv

# You'll see: "Skipping already processed ID: 95"
```

#### **Use Different LLM Provider**

```bash
# Use Groq (fast, free, requires API key):
python main.py --provider groq --dataset train.csv --output groq_results.csv

# Use Ollama (local, requires Ollama installed and running):
python main.py --provider ollama --dataset train.csv --output ollama_results.csv

# Use Claude (requires API key and credits):
python main.py --provider anthropic --dataset test.csv --output claude_submission.csv

# Use GPT-4 (requires API key and credits):
python main.py --provider openai --dataset test.csv --output gpt4_submission.csv

# Use Google Gemini (requires API key):
python main.py --provider google --dataset test.csv --output gemini_submission.csv
```

**Notes**:

- The `--provider` flag works for: groq, ollama, anthropic, openai, google
- For HuggingFace or DeepSeek, edit `llm_provider` in `config.yaml`
- Default provider (HuggingFace) is used if no `--provider` flag given

#### **Single Narrative Mode** (Custom Input)

Test on your own narrative and backstory:

```bash
python main.py \
  --narrative data/The\ Count\ of\ Monte\ Cristo.txt \
  --backstory my_backstory.txt \
  --output single_result.csv

# Creates a CSV with one prediction
```

## 📊 Configuration

Edit `config.yaml` to customize:

```yaml
# Primary LLM provider
llm_provider: "huggingface" # huggingface, deepseek, groq, ollama, anthropic, openai, google

# HuggingFace Configuration (FREE - default)
providers:
  huggingface:
    model: "meta-llama/Meta-Llama-3-8B-Instruct"
    temperature: 0.1
    max_tokens: 300

  groq:
    model: "llama-3.1-8b-instant" # Fast and efficient
    temperature: 0.1
    max_tokens: 300

  ollama:
    model: "phi3:mini" # Only needs ~2GB RAM
    temperature: 0.1
    num_ctx: 2048
    max_tokens: 300

# Self-consistency
self_consistency:
  enabled: true
  num_chains: 2 # Number of reasoning chains (can increase for higher accuracy)
  voting_strategy: "weighted" # or "majority"
  early_stop_confidence: 0.85 # Stop early when confident

# Multi-agent system (optional - disabled by default for speed)
multi_agent:
  enabled: false
  deliberation_rounds: 1

# Ensemble (optional - disabled by default for speed)
ensemble:
  enabled: false
  models:
    - provider: "groq"
      model: "llama-3.1-8b-instant"
      weight: 1.0
```

## 💰 Cost Analysis

| Configuration                        | Accuracy | Cost          | Speed  |
| ------------------------------------ | -------- | ------------- | ------ |
| **HuggingFace Llama 3 8B (default)** | ~60%     | **$0** (FREE) | Fast   |
| **+ Fallback (Groq/Ollama)**         | ~65%     | **$0** (FREE) | Fast   |
| **+ Self-consistency (2 chains)**    | ~70%     | **$0** (FREE) | Medium |
| **+ More chains (5-10)**             | ~75-80%  | **$0** (FREE) | Slower |
| **+ Multi-agent**                    | ~80-85%  | **$0** (FREE) | Slower |
| **+ Claude 3.5 Sonnet**              | ~90%+    | ~$20-50       | Fast   |
| **+ Ensemble (3 models)**            | ~92%+    | ~$30-70       | Slower |

_All processing is FREE when using HuggingFace/Groq/DeepSeek/Ollama - no credit card required!_

**Note**: Accuracy estimates are approximate and depend on dataset characteristics, prompt tuning, and model availability.

## 🔧 Advanced Usage

### Disable Specific Features (Faster Processing)

```bash
# Disable self-consistency (faster, lower accuracy)
python main.py --no-self-consistency --dataset train.csv

# Disable multi-agent (faster, lower accuracy)
python main.py --no-multi-agent --dataset train.csv

# Disable reranker (faster)
python main.py --no-reranker --dataset train.csv

# Speed mode: disable all advanced features
python main.py --no-self-consistency --no-multi-agent --no-reranker --dataset train.csv
```

### Custom Configuration

```bash
python main.py --config custom_config.yaml --dataset train.csv
```

### Debug Mode

```bash
python main.py --log-level DEBUG --dataset train.csv
```

### Process Specific Provider

```bash
# Override config file provider (supported: groq, ollama, anthropic, openai, google)
python main.py --provider groq --dataset train.csv
python main.py --provider anthropic --dataset test.csv

# Note: To use HuggingFace or DeepSeek, edit config.yaml and set llm_provider
```

### **Common Issues During Installation/Running**

#### "Command 'python' not found"

```bash
# Use python3 instead of python:
python3 main.py --dataset train.csv

# Or create an alias:
alias python=python3
```

#### "ModuleNotFoundError: No module named 'pathway'"

```bash
# Make sure virtual environment is activated:
source venv/bin/activate

# Reinstall requirements:
pip install -r requirements.txt
```

#### "No .env file found" or "API key not found"

```bash
# Create .env from template:
cp .env.example .env

# Edit it and add your keys:
nano .env

# Verify it exists:
cat .env
```

#### Logs show "Rate limit exceeded"

- This is normal! The fallback system will automatically switch providers
- Make sure you have multiple API keys set up
- Or wait a few minutes and the rate limit will reset

#### "Can't find train.csv or test.csv"

```bash
# Make sure you're in the project root directory:
pwd
# Should show: .../hackathon

ls -la
# Should see: train.csv, test.csv, main.py, config.yaml

# If missing, download from hackathon materials
```

---

## 📁 Project Structure

```
hackathon/                          # Root project directory
├── 📄 main.py                      # CLI entry point - run this file
├── 📄 config.yaml                  # Configuration (LLM models, settings)
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Template for API keys
├── 📄 .env                         # Your actual API keys (gitignored)
├── 📄 .gitignore                   # Git ignore rules
├── 📄 README.md                    # This file
├── 📄 SETUP.md                     # Detailed setup instructions
├── 📄 USAGE.md                     # Usage examples
├── 📄 DATASET_GUIDE.md             # Dataset structure explanation
│
├── 📊 train.csv                    # Training data (140 examples)
├── 📊 test.csv                     # Test data (59 examples to predict)
│
├── 📁 data/                        # Narrative text files (novels)
│   ├── In search of the castaways.txt    # Full novel (~100k words)
│   └── The Count of Monte Cristo.txt     # Full novel (~100k words)
│
├── 📁 src/                         # Source code modules
│   ├── 📄 __init__.py              # Package initializer
│   ├── 📄 config.py                # Configuration loader
│   ├── 📄 llm_providers.py         # LLM abstraction (Groq, Ollama, etc.)
│   ├── 📄 pathway_ingestion.py    # Pathway vector store & chunking
│   ├── 📄 self_consistency.py     # Self-consistency engine (10 chains)
│   ├── 📄 multi_agent.py           # Multi-agent system (4 agents)
│   ├── 📄 ensemble.py              # Ensemble voting logic
│   └── 📄 pipeline.py              # Main processing pipeline
│
├── 📁 logs/                        # Log files (auto-created)
│   └── narrator_*.log              # Timestamped log files
│
└── 📁 venv/                        # Virtual environment (gitignored)
    └── ...                         # Python packages
```

### 📊 Dataset Structure

**train.csv** (140 labeled examples):

```csv
id,book_name,char,caption,content,label
46,In Search of the Castaways,Thalcave,,"Backstory text...",consistent
137,The Count of Monte Cristo,Faria,...,"Backstory text...",contradict
```

**test.csv** (59 unlabeled examples for submission):

```csv
id,book_name,char,caption,content
95,The Count of Monte Cristo,Noirtier,,"Backstory text..."
136,The Count of Monte Cristo,Faria,,"Backstory text..."
```

**Output Format** (what you submit):

```csv
Story ID,Prediction,Rationale
95,0,"Backstory contradicts established timeline in Chapter 15..."
136,1,"Backstory aligns with character development arc..."
```

Where:

- `Prediction`: **1** = Consistent, **0** = Inconsistent
- `Rationale`: Brief explanation (1-2 sentences)

## 🎓 How It Works

### 1. **Data Loading** (CSV-based)

The system reads `train.csv` or `test.csv` which contains:

- Character backstories (in the `content` column)
- Book names (maps to novel files in `data/`)
- Example IDs

For each row, it loads the corresponding full novel from `data/` folder.

### 2. **Pathway Integration** (Required for Track A)

- Uses Pathway's vector store for semantic search
- Semantic chunking preserves narrative coherence
- Hybrid search (semantic + keyword) for comprehensive retrieval
- BGE-large embeddings for high-quality representations

### 3. **Self-Consistency Reasoning**

Generates multiple independent reasoning chains (default: 2, configurable up to 10+) with different strategies:

- **Direct Analysis**: Straightforward consistency check
- **Timeline Reconstruction**: Temporal ordering validation
- **Character Psychology**: Personality and motivation alignment
- **Causal Chain Validation**: Cause-effect relationships
- **Contradiction Search**: Active search for conflicts
- **Socratic Questioning**: Critical questioning approach
- **Devil's Advocate** (2 versions): Argue both sides
- **Step-by-Step Logic**: Formal logical analysis
- **Counterfactual Reasoning**: "What if" scenarios

Each chain votes independently with early stopping when confidence threshold (0.85) is reached. Results are aggregated using weighted or majority voting.

### 4. **Multi-Agent Adversarial System**

Four specialized agents collaborate:

- **Prosecutor**: Builds case for INCONSISTENCY
- **Defender**: Builds case for CONSISTENCY
- **Investigator**: Neutral fact-gathering
- **Judge**: Final decision based on all arguments

Agents deliberate over multiple rounds, refining their arguments.

### 5. **Ensemble Voting**

Aggregates predictions using:

- **Majority voting**: Equal weight to each prediction
- **Weighted voting**: Based on confidence scores
- **Soft voting**: Continuous probability scores

Final decision combines all reasoning methods for maximum accuracy.

## 🔬 Why This Approach Wins

### ✅ **Novel NLP Approach**

- Goes beyond basic RAG with adversarial reasoning
- Self-consistency with 10 diverse reasoning strategies
- Causal hypothesis generation and testing

### ✅ **Robust Long-Context Handling**

- Semantic chunking preserves narrative structure
- Multi-query retrieval captures different aspects
- Cross-encoder reranking improves precision

### ✅ **Evidence-Grounded**

- Every decision backed by specific text passages
- Multiple agents verify evidence independently
- Transparent reasoning chain

### ✅ **Pathway Integration**

- Meaningfully uses Pathway for vector store
- Efficient ingestion and retrieval
- Scalable to large narratives

### ✅ **Reproducible & Configurable**

- Works with FREE models (Groq, Ollama)
- Easy to upgrade to paid models
- Clear documentation for judges to run

## 🐛 Troubleshooting

### Windows: "Pathway not found" or Import Errors

**Solution**: Pathway requires Linux. Use WSL:

```powershell
# Install WSL if not already installed
wsl --install

# Start WSL
wsl

# Navigate to project (Windows D: drive maps to /mnt/d/)
cd /mnt/d/Learning/Kharapur_Hackathon/hackathon

# Run from WSL
python main.py --dataset train.csv --output results.csv
```

### "API key not found" errors

```bash
# Make sure .env file exists with your actual keys:
HUGGINGFACE_API_KEY=hf_your_actual_key_here
GROQ_API_KEY=gsk_your_actual_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Check if file exists
cat .env  # Linux/WSL
type .env  # Windows CMD
```

### "Rate limit" or "402 Payment Required" errors

The system automatically falls back through providers:

- **HuggingFace → Groq → Ollama**

Make sure you have multiple API keys set up, or install Ollama as a final fallback:

```bash
# Install Ollama: https://ollama.ai
ollama pull phi3:mini
```

### "Ollama connection failed"

```bash
# Install Ollama first: https://ollama.ai
ollama serve  # Start server (run in background)
ollama pull phi3:mini  # Download model (~2GB)
# or
ollama pull llama3.1:8b  # Larger model (~4.7GB)
```

### "Out of memory" errors

```yaml
# Edit config.yaml - reduce chunk size:
pathway:
  chunking:
    chunk_size: 500 # Smaller chunks

# Or reduce reasoning chains:
self_consistency:
  num_chains: 1 # Single chain (fastest)
```

### WSL: "Permission denied" errors

```bash
# Fix permissions
chmod +x main.py
chmod -R 755 src/
```

### Dataset not found

```bash
# Make sure files are in correct location:
ls -la  # Should show train.csv, test.csv
ls data/  # Should show two .txt files

# If missing, download from hackathon Google Drive
```

## 📝 Output Format

Results are saved in CSV format:

```csv
Story ID,Prediction,Rationale
1,1,"The backstory is consistent because character's motivations align with actions..."
2,0,"Inconsistent due to timeline conflict: backstory states X but narrative shows Y..."
```

- **Prediction**: `1` = Consistent, `0` = Inconsistent
- **Rationale**: Brief explanation (1-2 sentences)

## 🎯 Performance Tips

### For Maximum Accuracy:

1. Enable all features (self-consistency + multi-agent + ensemble)
2. Set multiple API keys for fallback reliability
3. Increase `num_chains` to 10-20 in config.yaml
4. Enable multi-agent: set `multi_agent.enabled: true`
5. If budget allows, add Claude 3.5 to ensemble or use as primary

### For Maximum Speed:

1. Disable multi-agent: `--no-multi-agent` (already disabled by default)
2. Reduce chains: set `num_chains: 1`
3. Use fast model: keep default `llama-3.1-8b-instant` or `phi3:mini`
4. Disable reranker: `--no-reranker`
5. Set `early_stop_confidence: 0.75` for faster decisions

## 📄 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- **Pathway** for the powerful data processing framework
- **Groq** for free, fast LLM inference
- **Ollama** for local model deployment
- Hackathon organizers for this challenging problem!

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the team.

---

**Built with ❤️ for Kharagpur Data Science Hackathon 2026**
