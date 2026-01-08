# Ouroboros-Recursive-LM
# Recursive Reasoning LLM (R-HRM)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

A highly efficient, adaptable Large Language Model architecture designed to democratize AI training. by utilizing **Recursive Layers** and **Hierarchical Reasoning**, this model achieves deep logic capabilities with a fraction of the parameters found in standard Transformers. It is capable of training on consumer hardware (even CPUs) while scaling up seamlessly to GPUs.

## 🧠 Core Architecture

The **R-HRM** (Recursive Hierarchical Reasoning Model) diverges from traditional architectures by focusing on *depth through recursion* rather than simple layer stacking.

### Key Components

1.  **Recursive Transformer Blocks**:
    *   Instead of having 12 distinct layers, the model uses a **single shared block** and feeds the data through it recursively (4-8 times).
    *   **Benefit**: Massive parameter reduction (10x smaller footprint) while maintaining the non-linear "reasoning depth" of a much larger model.
    *   **Brain Pruning**: Uses aggressive `Weight Decay` (0.1) to force the recursive weights to learn generalizable rules rather than memorizing data.

2.  **Multi-Head Latent Attention (MLA)**:
    *   Optimizes memory bandwidth by compressing Key-Value pairs into latent vectors (`d_latent`) before projection.
    *   **Benefit**: Faster inference and lower VRAM usage during generation.

3.  **Modern Enhancements**:
    *   **RMSNorm**: Replaces LayerNorm for superior gradient stability.
    *   **SwiGLU**: A gated activation function that allows the Feed-Forward Network to select which information to pass through, mimicking biological gating.
    *   **Rotary Embeddings (RoPE-compatible logic)**: Implicitly handled via position-aware projections.

## 🚀 Usage

### 1. Installation
Install the required dependencies:
```bash
pip install torch transformers datasets huggingface_hub
```

### 2. Training
The training script is **Hardware-Aware**. It automatically detects if you are running on a local CPU or a Cloud GPU (e.g., Colab, A100) and switches configurations instantly.

```bash
python train.py
```

| Mode | Detects | Config | Description |
| :--- | :--- | :--- | :--- |
| **Lite** | CPU | `B=4`, `L=4`, `D=256` | Optimized for low-memory local testing. |
| **Pro** | GPU (CUDA) | `B=16`, `L=8`, `D=512` | High-performance mode with **Mixed Precision (AMP)** and bigger batches. |

**Training Features:**
*   **Infinite Streaming**: Never runs out of data (auto-restarts iterators).
*   **Safety Net**: Validation loops are crash-proof.
*   **Sniper Stop**: Automatically stops training when Validation Loss hits **2.1** (The "Reasoning Convergence" point).

### 3. Chatting (Inference)
Interact with your trained model. The script automatically reads the checkpoint to determine the correct model size.

```bash
python chat.py
```

**Generation Features:**
*   **N-Gram Blocking**: Prevents repetitive loops (e.g., "(((").
*   **Nucleus Sampling (Top-P)**: sets `p=0.9` to ensure creative but coherent responses.
*   **Repetition Penalty**: actively discourages reusing recent words.

## 📚 Data Pipeline

The model trains on a curated, interleaved stream of high-quality datasets (~5GB equivalent):
1.  **TinyGSM** (40%): Grade-school math reasoning.
2.  **TinyCodes** (30%): Python/programming logic.
3.  **Cosmopedia** (15%): Synthetic textbooks for scientific knowledge.
4.  **MetaMathQA** (15%): Advanced mathematical problems.

**Filters**:
*   **Symbol Spam Blocking**: Automatically rejects lines with >10 consecutive symbols to prevent "glitch tokens".
*   **Length Cap**: Filters super-long sequences to maintain context focus.

## 📂 File Structure

*   `model.py`: The PyTorch definition of the Recursive HRM architecture.
*   `train.py`: Main training loop with device detection, scheduling, and accumulation.
*   `data_loader.py`: Handles streaming, interleaving, and rigorous data cleaning.
*   `chat.py`: Inference interface with robust generation logic.

## 🏆 Credits & Inspirations

This project stands on the shoulders of giants. The architecture integrates breakthrough concepts from top-tier AI research:

*   **Recursive Layers**: Inspired by **ALBERT** (Google Research) and **Universal Transformers**, proving that weight sharing can achieve high performance with efficient parameter usage.
*   **Multi-Head Latent Attention (MLA)**: A technique popularized by **DeepSeek-V2**, significantly compressing memory usage (KV Cache) without sacrificing long-context retrieval.
*   **SwiGLU Activation**: The "Gated Linear Unit" variant proposed by **Shazeer et al. (2020)** and used in **Meta's Llama** and **Google's PaLM**, offering superior convergence over RELU/GELU.
*   **RMSNorm**: A stabilizing normalization technique (Zhang & Sennrich, 2019) that is now the industry standard for LLMs like **Llama 3** and **Gemma**.
*   **Datasets**:
    *   *TinyGSM* (Microsoft Research/Phi-1 logic)
    *   *Cosmopedia* (HuggingFace H4 Team)
    *   *TinyCodes* (nampdn-ai)

---
*Built for the Future of Efficient AI.*
