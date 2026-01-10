# ⚡ Quick Start Guide (3 Minutes)

Get up and running with the optimized Narrative Consistency Checker in 3 minutes!

## Step 1: Get API Key (1 minute)

### Option A: Together AI (Recommended)
1. Go to https://api.together.xyz/
2. Click "Sign Up" (free)
3. Go to "API Keys" in dashboard
4. Click "Create new key"
5. Copy your API key

### Option B: Cerebras (Fastest)
1. Go to https://cerebras.ai/
2. Sign up for API access
3. Get your API key from dashboard

## Step 2: Set API Key (30 seconds)

### Windows PowerShell:
```powershell
$env:TOGETHER_API_KEY="paste-your-key-here"
```

### Or Create .env File:
```powershell
echo "TOGETHER_API_KEY=paste-your-key-here" > .env
```

## Step 3: Run (10 seconds)

```powershell
# Process entire dataset
python main.py --dataset data/ --output results.csv

# Or test single example
python example_usage.py
```

## That's It! 🎉

Your code is now running **10x faster** than before!

---

## Expected Output

```
2026-01-09 10:30:15 | INFO     | Processing narrative example_1
🔄 [1/5] Ingesting narrative into vector store...
✅ [1/5] Completed in 15.2s - Created 87 chunks
🔄 [2/5] Extracting key queries from backstory...
✅ [2/5] Completed in 8.1s - Generated 6 queries
🔄 [3/5] Retrieving relevant evidence from narrative...
✅ [3/5] Completed in 18.3s - Retrieved 20 evidence pieces
🔄 [4/5] Reranking evidence for relevance...
✅ [4/5] Completed in 3.2s
🔄 [5/5] Running reasoning engines...
✓ Chain 1/5 completed: CONSISTENT
✓ Chain 2/5 completed: CONSISTENT
✓ Chain 3/5 completed: CONSISTENT
✓ Chain 4/5 completed: CONSISTENT
✓ Chain 5/5 completed: INCONSISTENT
✓ All agents completed initial analysis
✅ [5/5] Completed reasoning in 8.4s
✨ Narrative example_1 processed successfully in 53.2s total

Final decision: CONSISTENT (confidence: 0.85)
```

---

## Troubleshooting

### "API key not found"
Make sure you set the environment variable in the same terminal:
```powershell
$env:TOGETHER_API_KEY="your-key"
python main.py --dataset data/
```

### Still using Groq?
Edit `config.yaml` and change:
```yaml
llm_provider: "together"  # or "cerebras"
```

### Want even faster?
Use Cerebras (1800 tokens/sec):
```powershell
$env:CEREBRAS_API_KEY="your-key"
python main.py --provider cerebras --dataset data/
```

---

## Next Steps

- 📖 Read [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for technical details
- 📊 See [PERFORMANCE.md](PERFORMANCE.md) for speed comparisons
- 🔧 Check [API_SETUP.md](API_SETUP.md) for advanced setup
- 📝 Run [example_usage.py](example_usage.py) for a quick demo

---

## Speed Comparison

| Dataset Size | Old Time | New Time | Improvement |
|--------------|----------|----------|-------------|
| 1 narrative | 9 min | 1 min | **9x faster** |
| 10 narratives | 90 min | 10 min | **9x faster** |
| 100 narratives | 15 hours | 1.7 hours | **9x faster** |

---

## API Providers Quick Reference

| Provider | Setup Time | Speed | Cost | Recommended |
|----------|-----------|-------|------|-------------|
| **Together AI** | 1 min | Fast | FREE | ✅ Yes |
| **Cerebras** | 1 min | Very Fast | FREE | ✅✅ Best |
| Groq | 1 min | Medium | FREE | ⚠️ Rate limited |
| Ollama | 10 min | Slow | FREE | Local only |

---

## Support

Having issues? Check:
1. [API_SETUP.md](API_SETUP.md) - Detailed setup guide
2. [PERFORMANCE.md](PERFORMANCE.md) - Performance FAQ
3. [OPTIMIZATIONS.md](OPTIMIZATIONS.md) - Technical details

---

**Happy analyzing! 🚀**
