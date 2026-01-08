# Dataset Structure Guide

## 📁 Current Folder Structure

```
D:\Learning\Kharapur_Hackathon\hackathon\
├── data/                                    # Narrative text files
│   ├── In search of the castaways.txt       # Full novel (100k+ words)
│   └── The Count of Monte Cristo.txt        # Full novel (100k+ words)
│
├── train.csv                                # Training data (140 examples)
├── test.csv                                 # Test data (59 examples)
│
├── main.py                                  # Main entry point
├── config.yaml                              # Configuration
├── narrator.py
├── requirements.txt
├── src/                                     # Source code
│   ├── __init__.py
│   ├── config.py
│   ├── ensemble.py
│   ├── llm_providers.py
│   ├── multi_agent.py
│   ├── pathway_ingestion.py
│   ├── pipeline.py
│   └── self_consistency.py
│
└── venv/                                    # Virtual environment (WSL)
```

## 📊 CSV File Structure

### **train.csv** (140 rows)

| Column      | Description         | Example                      |
| ----------- | ------------------- | ---------------------------- |
| `id`        | Unique example ID   | 46, 137, 74...               |
| `book_name` | Novel name          | "In Search of the Castaways" |
| `char`      | Character name      | "Thalcave", "Faria"          |
| `caption`   | Optional context    | "" or "The Origin of..."     |
| `content`   | Character backstory | "Thalcave's people faded..." |
| `label`     | Ground truth        | "consistent" or "contradict" |

### **test.csv** (59 rows)

Same structure as `train.csv` but **without the `label` column**.

## 🎯 How It Works

1. **Narrative (Novel)**: The full 100k+ word novel text
2. **Backstory (Content)**: A hypothetical character background (from CSV)
3. **Task**: Determine if the backstory is consistent (1) or contradictory (0) with the novel

### Example:

- **Novel**: "In Search of the Castaways" (100k+ words)
- **Backstory**: "Thalcave's people faded as colonists advanced..."
- **Label**: "consistent" → means this backstory fits the novel
- **Prediction**: System should output **1**

## 🚀 How to Run

### **Train on training data:**

```bash
python main.py --dataset train.csv --output train_results.csv
```

### **Generate predictions for test data:**

```bash
python main.py --dataset test.csv --output test_predictions.csv
```

### **Process single example (for testing):**

```bash
python main.py --dataset test.csv --output results.csv
```

## 📝 Output Format

The program generates a CSV file with:

- `Story ID`: Example ID from input CSV
- `Prediction`: 0 (inconsistent) or 1 (consistent)
- `Rationale`: Brief explanation (optional for Track A)

### Example Output:

```csv
Story ID,Prediction,Rationale
46,1,Backstory aligns with narrative timeline and character development
137,0,Direct contradiction with established facts in Chapter 15
74,1,Consistent with cultural practices described in the novel
```

## 🔄 Data Flow

```
train.csv/test.csv
      ↓
Load backstories from CSV rows
      ↓
Load corresponding novel from data/*.txt
      ↓
Process each (narrative, backstory) pair
      ↓
[Vector Store] → [Self-Consistency] → [Multi-Agent] → [Ensemble]
      ↓
Output: Prediction (0 or 1) + Confidence + Reasoning
      ↓
Save to results CSV
```

## 🎓 Label Mapping

In the code:

- `"consistent"` → **1**
- `"contradict"` → **0**

This matches the hackathon submission format requirement.
