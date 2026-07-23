[README.md](https://github.com/user-attachments/files/30327196/README.md)
# Neural Networks: Zero to Hero — My Implementation

Personal walkthrough of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series, built from scratch rather than copied — every derivation is worked out by hand before being implemented in code.

## Why this repo exists

I'm using this series to build a solid foundation before starting a research-oriented exchange year, where I'll be working on generative models (diffusion) and uncertainty quantification. The goal here isn't to use PyTorch as a black box — it's to understand exactly what it's doing underneath: autograd, backpropagation, and the building blocks that later become full architectures like GPT.

## What I'm covering

### Autograd from scratch (micrograd)
Building a scalar-valued automatic differentiation engine with no framework at all. Covers the computational graph, the chain rule applied node by node, and implementing `backward()` manually for operations like addition, multiplication, `tanh`, and `exp`. This is the part that makes PyTorch stop being magic.

### Character-level language modeling (makemore)
A progression through several architectures, all trained on the same task (generating name-like words), which makes it easy to see what each added layer of complexity actually buys you:
- **Bigram model**: pure counting and probability, the simplest possible language model.
- **MLP**: introduces embeddings, hidden layers, and batching.
- **Manual backpropagation**: deriving every gradient by hand instead of relying on autograd, to really internalize what's happening at each layer.
- **WaveNet-style architecture**: hierarchical, dilated structure for going deeper without losing signal.

### Building GPT from scratch (nanoGPT)
Implementing the Transformer architecture piece by piece: self-attention, multi-head attention, positional encoding, layer normalization, and the full training loop — ending with a small GPT trained on character-level text.

### Tokenization (minBPE)
Implementing Byte Pair Encoding from scratch to understand how modern LLMs actually turn text into tokens, and comparing the result against OpenAI's tokenizer.

## Core concepts I'm making sure I actually understand (not just run)

- Computational graphs and reverse-mode automatic differentiation
- The chain rule as the engine behind every gradient update
- Why certain derivatives (like `exp`) have particularly clean backward passes
- Embeddings and how they're learned rather than hand-designed
- Self-attention as a communication mechanism between tokens
- The training loop end to end: forward pass, loss, backward pass, parameter update

## Environment

```bash
conda create -n karpathy-zero-to-hero python=3.10 -y
conda activate karpathy-zero-to-hero
conda install -c pytorch pytorch torchvision torchaudio -y
conda install numpy matplotlib jupyter notebook ipykernel -y
conda install -c conda-forge graphviz python-graphviz -y
pip install requests tiktoken
```

## Where this is heading

Once this series is done, the next step is implementing a **Denoising Diffusion Probabilistic Model (DDPM) from scratch**, then extending it with an uncertainty estimation layer — the bridge project toward the research I want to pursue during my exchange year (uncertainty quantification in generative models).

## Reference

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — Andrej Karpathy
