<h1 align="center">🧠 LLM Decoding Dynamics & Preference Alignment 🤖</h1>
<p align="center">
  <b>Comparative Analysis of Generation Strategies & DPO-based Policy Alignment</b><br>
  <i>A Research Framework for Investigating Diversity-Quality Trade-offs and Alignment Stability</i>
</p>

---

## 📌 Project Overview

This project provides a rigorous technical exploration into the post-training lifecycle of Large Language Models (LLMs). We investigate how specific **Decoding Strategies** influence lexical diversity and how **Direct Preference Optimization (DPO)** can align model behavior with human preferences while minimizing the "Alignment Tax."

The research is divided into three core pillars:
1. **Decoding Dynamics**: Benchmarking Greedy, Beam Search, Top-K, and Top-P strategies.
2. **Preference Alignment**: Implementing and evaluating **DPO** and **GRPO** for robust policy refinement.
3. **Mechanistic Interpretability**: Using **Universal Sparse Autoencoders (USAEs)** to test the Platonic Representation Hypothesis.

---

## 🧑‍💻 My Contributions

In this collaborative research effort, I served as the **Lead Researcher for Decoding Dynamics and DPO Implementation**. My specific technical contributions included:

- **End-to-End Implementation of Task 1**: Developed the sequential generation loops for all four decoding strategies and designed the evaluation suite for $Distinct-N$ diversity metrics.
- **DPO Pipeline Orchestration (Task 2)**: Integrated **LoRA (Low-Rank Adaptation)** for parameter-efficient alignment of the `SmolLM2-135M` model. 
- **Robustness Analysis**: Conducted the empirical study on **Catastrophic Forgetting** and **Verbosity Bias**, specifically calculating the KL-Divergence and Perplexity shifts for the DPO-aligned policy.

---

## 🧠 Architecture & Methodology

### 🔹 Module 1: Decoding Strategy Analysis
We analyzed the trade-offs between **Search-based** (Greedy, Beam) and **Sampling-based** (Top-K, Top-P) methods.
- **Objective**: Balance generation quality (Perplexity) with lexical variety ($Distinct-1/2/3$).
- **Key Insight**: Top-P (Nucleus) sampling consistently outperformed Top-K by dynamically adjusting the sampling pool based on the cumulative probability mass, preventing "boring" or repetitive loops common in Beam Search.



---

### 🔹 Module 2: LLM Alignment (DPO vs. GRPO)
Using the `orca_dpo_pairs` dataset, we optimized a Supervised Fine-Tuned (SFT) model using Direct Preference Optimization.
- **Optimization Strategy**: Utilized LoRA ($r=8, \alpha=16$) to fine-tune only the Query/Value projections, significantly reducing the compute footprint.
- **Comparison**: While **GRPO** showed signs of mode collapse and reward hacking, **DPO** remained stable, effectively increasing the likelihood of preferred responses without significantly degrading general language capabilities.



---

### 🔹 Module 3: Mechanistic Interpretability (USAEs)
We investigated the **Platonic Representation Hypothesis** by training Universal Sparse Autoencoders.
- **Goal**: Align the internal feature spaces of different architectures (ResNet vs. ViT).
- **Finding**: Despite different structural priors, both models converged toward a shared semantic "Platonic" space, though alignment accuracy was inversely correlated with reconstruction loss.

---

## 📊 Results Summary

| Strategy/Method | Perplexity ($\downarrow$) | $Distinct-3$ ($\uparrow$) | KL-Div ($\approx 0$) | Status |
|-----------------|---------------------------|----------------------------|-----------------------|--------|
| **SFT (Baseline)** | 10.23 | 0.84 | 0.00 | Reference |
| **Top-P (p=0.9)** | 10.55 | **0.91** | - | Best Diversity |
| **DPO Alignment** | **11.00** | 0.87 | **0.016** | **Stable Alignment** |
| **GRPO Alignment**| 14.20 | 0.72 | 0.082 | Mode Collapse |

---

## 📂 Project Structure
```text
llm-decoding-preference-alignment/
├── Task1_Decoding/
│   └── llm_decoding_strategies_and_diversity_metrics.py  # Implementation of Search/Sampling
├── Task2_Alignment/
│   ├── dpo_alignment_pipeline.py                         # My DPO & Robustness Implementation
│   └── grpo_analysis_baseline.py                         # GRPO Comparative Baseline
├── Task3_Interpretability/
│   └── universal_sae_alignment.py                        # USAE & Feature Correlation
├── docs/
│   └── Technical_Project_Report.pdf                      # Comprehensive ICML-style Paper
└── README.md
