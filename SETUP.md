# Setup Instructions

## Quick Setup (5 minutes)

### 1. Install Python Dependencies

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
```

### 2. Get FREE Groq API Key

1. Go to https://console.groq.com
2. Sign up (free, no credit card required)
3. Create an API key
4. Copy the key

### 3. Configure Environment

```powershell
# Copy example environment file
copy .env.example .env

# Edit .env and add your Groq API key
notepad .env
```

Add this line to `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Test Installation

```powershell
# Test with a simple example
python -c "from src.config import get_config; print('✓ Configuration loaded successfully')"
```

## Optional: Install Ollama (Local Models)

If you want to run models locally without any API calls:

1. Download Ollama from https://ollama.ai
2. Install and start Ollama
3. Pull a model:
   ```powershell
   ollama pull llama3.1:70b
   # Or smaller model:
   ollama pull llama3.1:8b
   ```
4. Update `config.yaml`:
   ```yaml
   llm_provider: "ollama"
   ```

## Optional: Premium Models Setup

### Claude (Anthropic)
```powershell
# Add to .env
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

### GPT-4 (OpenAI)
```powershell
# Add to .env
OPENAI_API_KEY=sk-your_key_here
```

### Gemini (Google)
```powershell
# Add to .env
GOOGLE_API_KEY=your_key_here
```

## Verification

Run this to verify everything is working:

```powershell
python main.py --help
```

You should see the help message with all available options.

## Troubleshooting

### "ModuleNotFoundError: No module named 'pathway'"

```powershell
pip install pathway>=0.8.0
```

### "ModuleNotFoundError: No module named 'groq'"

```powershell
pip install groq>=0.4.0
```

### "API key not found"

Make sure:
1. `.env` file exists in the project root
2. `.env` contains `GROQ_API_KEY=your_key`
3. No extra spaces around the `=` sign

### GPU/CUDA Issues

If you don't have a GPU, edit `config.yaml`:
```yaml
embeddings:
  device: "cpu"  # Change from "cuda" to "cpu"
```

## Next Steps

Once setup is complete:

1. **Download the dataset** from the hackathon organizers
2. **Place it in a `data/` folder**
3. **Run the system**: `python main.py --dataset data/ --output results.csv`

See `README.md` for detailed usage instructions.
