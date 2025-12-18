
# The Thermodynamics of Truth: Topological Entropy as a Universal Coherence Metric in LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Paper Status](https://img.shields.io/badge/Status-Preprint-green)](https://zenodo.org/)

**Official repository for the paper:** *"The Thermodynamics of Truth: Topological Entropy as a Universal Coherence Metric in Large Language Models"*

**Author:** Andrés Sebastián Pirolo  
**Contact:** apirolo@abc.gob.ar

---

## 🧪 Abstract

The detection of hallucinations in Large Language Models (LLMs) remains a critical challenge for AI safety. This research introduces the **Topological Entropy Index (TEI)**, a novel unsupervised method that analyzes the internal geometry of the self-attention mechanism.

By treating attention matrices as directed graphs and calculating the Shannon entropy of their PageRank distribution, we reveal a **monotonic entropy gradient** governing artificial cognition. Our experiments with **TinyLlama-1.1B (N=1000 samples per condition)** demonstrate that:

* 💎 **Truth (Factual):** Low Entropy ($\mu=2.63$ bits) $\rightarrow$ Ordered "Crystal" Structure.
* 🔥 **Hallucination:** Medium Entropy ($\mu=2.99$ bits) $\rightarrow$ Thermodynamic Cost of Lying ($\Delta S = +0.36$).
* ⚡ **Noise:** Maximum Entropy ($\mu=4.68$ bits) $\rightarrow$ Topological Anarchy.

This metric allows for real-time, content-agnostic coherence monitoring without the need for external knowledge bases.

## 📊 Key Findings (N=1000)

We conducted a rigorous stress test using **N=1000 samples per condition** on an NVIDIA Tesla T4 GPU. The results confirm a linear relationship between topological disorder and lack of semantic grounding.

| Condition | Mean Entropy (bits) | 95% CI | Cohen's d (vs Factual) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Factual (Truth)** | **2.635** | [2.61, 2.66] | - | **Maximum Structural Efficiency** |
| Word Salad | 2.805 | [2.79, 2.82] | 0.53 | Attention Sink Collapse |
| **Hallucination** | **2.993** | [2.97, 3.02] | **0.95** | **Intrinsic Cost of Fabrication** |
| Pure Noise | 4.681 | [4.66, 4.70] | 5.78 | Total Topological Collapse |

## 🚀 Installation

To replicate the experiments, clone this repository and install the dependencies. A GPU is recommended (Tesla T4 or better) for the N=1000 generation loop.

```bash
git clone [https://github.com/pirolo/topological-entropy-hallucination.git](https://github.com/pirolo/topological-entropy-hallucination.git)
cd topological-entropy-hallucination
pip install -r requirements.txt

Requirements
 * torch
 * transformers
 * networkx
 * numpy
 * pandas
 * scipy
 * matplotlib
 * seaborn
 * tqdm
💻 Usage
1. Run the Experiment (Generation & Analysis)
To generate the dataset and calculate entropy for all conditions:
python run_experiment.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --samples 1000 --gpu

2. Generate Figures
To reproduce the figures used in the paper (The Entropy Gradient and Topological Visualization):
python generate_figures.py

📂 Repository Structure
 * src/tei_metric.py: Core implementation of the Topological Entropy Index algorithm.
 * run_experiment.py: Main script for data generation and batch processing.
 * generate_figures.py: Plotting scripts for publication-quality images (Nature style).
 * data/results_n1000.csv: The raw dataset used in the paper.
 * figures/: Generated PDF/PNG outputs.
🛠️ The Algorithm (TEI)
The core logic relies on extracting the attention matrix A from the final layer, applying a sparsification threshold (\tau=0.05), and computing the Shannon entropy of the PageRank vector P.
import networkx as nx
import numpy as np

def calculate_tei(attention_matrix, threshold=0.05):
    # 1. Build Graph
    G = nx.DiGraph()
    rows, cols = np.where(attention_matrix > threshold)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    
    # 2. Compute PageRank
    try:
        centrality = nx.pagerank(G)
        probs = np.array(list(centrality.values()))
        
        # 3. Calculate Shannon Entropy
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        return entropy
    except:
        return 0.0

📜 Citation
If you use this code or methodology in your research, please cite the paper:
@article{pirolo2025thermodynamics,
  title={The Thermodynamics of Truth: Topological Entropy as a Universal Coherence Metric in Large Language Models},
  author={Pirolo, Andrés Sebastián},
  journal={Preprint},
  year={2025},
  url={[https://github.com/pirolo/topological-entropy-hallucination](https://github.com/pirolo/topological-entropy-hallucination)}
}

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
Research conducted with the assistance of advanced computational agents for data processing and validation.


