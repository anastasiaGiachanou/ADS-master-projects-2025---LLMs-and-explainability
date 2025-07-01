# Chain-of-Thought Reasoning Faithfulness Analysis

A comprehensive evaluation framework for measuring the faithfulness of Chain-of-Thought (CoT) reasoning in Large Language Models using neural entailment analysis across model scale.

## Overview

This project implements a novel approach to evaluate whether AI models' step-by-step explanations truly reflect their reasoning process. The framework compares traditional keyword-based methods with neural entailment analysis to measure causal relationships between reasoning steps.

## Key Features

- **Neural Entailment Analysis**: Uses DeBERTa-v3-MNLI to measure logical connections between reasoning steps
- **Comprehensive Faithfulness Evaluation**: Implements multiple metrics including early answering, robustness testing, and contextual relevance
- **Multi-Model Support**: Evaluates DeepSeek-R1-Distill-Qwen models (1.5B to 32B parameters)
- **Causal Graph Visualization**: Creates visual representations of reasoning chains with connection strengths
- **Reproducible Results**: Fixed sample indices ensure consistent evaluation across runs

## Quick Start

1. **Setup**: The notebook automatically installs required packages and loads the LogiQA dataset

2. **Model Selection**: Change the `MODEL_NAME` variable to test different model sizes:
   ```python
   MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Options: 1.5B, 7B, 14B, 32B
   ```

3. **Configuration**: Choose evaluation depth:
   ```python
   config = FULL_CONFIG    # Most comprehensive (4-5 min/sample)
   config = MEDIUM_CONFIG  # Balanced (2-3 min/sample)  
   config = FAST_CONFIG    # Quick evaluation (30-60 sec/sample)
   ```

4. **Run Evaluation**:
   ```python
   evaluation_results = evaluate_dataset(
       dataset['validation'],
       num_samples=50,
       config=config
   )
   ```

## Core Components

### Neural Causal Entailment Analyzer
The `CausalEntailmentAnalyzer` class uses DeBERTa-v3-MNLI to measure logical relationships:
- Converts reasoning step pairs into entailment tasks
- Computes causal scores based on entailment probability and contradiction absence
- Builds comprehensive causal matrices for reasoning chains

### Faithfulness Metrics
- **Logical Consistency**: Measures contradiction avoidance in reasoning steps
- **Contextual Relevance**: Evaluates semantic overlap with problem context
- **Conclusion Alignment**: Checks if reasoning leads to correct answers
- **Early Answering**: Tests if models can answer with partial reasoning
- **Robustness**: Measures stability when errors are introduced

### Causal Coherence Comparison
- **Keyword-based**: Traditional approach using explicit causal connectors
- **Neural-based**: Novel approach using entailment relationships
- **Graph Visualization**: Visual representation of reasoning structure

## Key Findings

The research reveals significant differences between evaluation methods:
- Keyword detection identifies causal language (scores: 0.339-0.391)
- Neural entailment tests actual logical relationships (scores: 0.077-0.098)
- Smaller models show negative correlation between reasoning quality and accuracy
- Larger models succeed through pattern recognition rather than complete logical deduction

## Output Files

- `evaluation_results.json`: Complete experimental data
- `comprehensive_report.txt`: Summary statistics and findings
- `causal_graph_sample_*.png`: Visual reasoning chain representations
- `*_comparison.png`: Metric comparison plots

## Usage Examples

### Analyze Individual Samples
```python
# Display detailed analysis of first result
display_detailed_example(evaluation_results, index=0)

# Build causal graph for specific sample
graph = analyze_causal_graph_for_sample(evaluation_results, sample_index=0)
```

### Compare Methods
```python
# Compare coherence measurement approaches
for result in evaluation_results[:5]:
    print(f"Keyword coherence: {result['cot_causal_coherence']:.3f}")
    print(f"Neural coherence: {result['cot_causal_coherence_neural']:.3f}")
```

## Customization

### Adding New Models
Modify the `MODEL_NAME` variable to evaluate different models:
```python
MODEL_NAME = "your-model-name-here"
```

### Adjusting Sample Size
Change the evaluation scope (max 50 with current fixed indices):
```python
NUM_SAMPLES = 25  # Reduce for faster testing
```

### Custom Evaluation Points
Modify truncation points for early answering analysis:
```python
CUSTOM_CONFIG = {
    'early_answering_points': [0, 0.33, 0.67, 1.0],
    'mistake_positions': [0, 1, -1]  # First, second, and last steps
}
```

## Technical Details

### Dataset
- **LogiQA**: Logical reasoning questions from Chinese Civil Service Exam
- **50 Fixed Samples**: Reproducible evaluation subset from validation set
- **Multiple Choice**: 4-option questions requiring multi-step inference

### Model Evaluation
- **Greedy Decoding**: Deterministic outputs for reproducibility
- **16-bit Precision**: Memory-efficient model loading
- **Batch Processing**: Efficient neural entailment analysis

## Installation and Setup

### Running in Google Colab
1. Open the notebook in Google Colab
2. Run the installation cell to install all dependencies
3. The notebook will automatically download the LogiQA dataset
4. Choose your model size and configuration
5. Execute the evaluation cells

### Local Setup
1. Use the notebook in the "VisualStudioCodeLocal" folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook in Vscode or convert to Python script
4. Ensure adequate GPU memory for larger models

## Performance Considerations

- **Memory Usage**: Larger models (32B) require significant GPU memory
- **Runtime**: Full evaluation takes 4-5 hours for 50 samples with FULL_CONFIG
- **Storage**: Results and graphs can consume several GB of disk space


## Related Work

This work builds upon several key papers in the field:
- Chain-of-thought prompting elicits reasoning in large language models (Wei et al., 2022)
- Language models don't always say what they think (Turpin et al., 2023)
- Measuring faithfulness in chain-of-thought reasoning (Lanham et al., 2023)

## Contact

Feel free to contact me at h.f.brouwer@uu.nl or https://github.com/oioi123

## Citations


- R1 models
```

@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}

```
- DebertaV3 model
```

@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
- LogiQA Dataset

```
@article{liu2020logiqa,
  title={Logiqa: A challenge dataset for machine reading comprehension with logical reasoning},
  author={Liu, Jian and Cui, Leyang and Liu, Hanmeng and Huang, Dandan and Wang, Yile and Zhang, Yue},
  journal={arXiv preprint arXiv:2007.08124},
  year={2020}
}
```
- MNLI Dataset

```
@InProceedings{N18-1101,
  author = "Williams, Adina
            and Nangia, Nikita
            and Bowman, Samuel",
  title = "A Broad-Coverage Challenge Corpus for
           Sentence Understanding through Inference",
  booktitle = "Proceedings of the 2018 Conference of
               the North American Chapter of the
               Association for Computational Linguistics:
               Human Language Technologies, Volume 1 (Long
               Papers)",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  pages = "1112--1122",
  location = "New Orleans, Louisiana",
  url = "http://aclweb.org/anthology/N18-1101"
}
```
