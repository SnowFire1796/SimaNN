# SimaNN

**SimaNN** is a from-scratch, CPU-first C++26 neural language-model engine. It starts with random weights and is deliberately **untrained**. You provide a plain-text corpus and train it locally.

## Architecture

The initial model is a byte-level Elman recurrent neural network (RNN):

- 256-token vocabulary: each possible byte value is a token; UTF-8 text can be streamed directly.
- 96 hidden units by default.
- Tanh recurrent state and softmax next-byte prediction.
- Truncated backpropagation through time (BPTT).
- Adam optimizer with global gradient clipping.
- Top-k and temperature sampling for generation.
- Binary checkpoints include weights and Adam moments.
- Training reads windows from disk; it does not load the entire corpus into RAM.
- A held-out final 10% of the corpus is used to report validation loss.

Default model size at hidden=96: **58,720 trainable parameters**.

This is intentionally a small model for low-end CPUs. It is not comparable to a pretrained modern LLM. Byte-level generation may emit malformed UTF-8, particularly before adequate training. A small RNN learns local statistical patterns and is not expected to become a capable assistant without a substantial, carefully prepared corpus and further architecture work.

## Build on Xubuntu

```bash
sudo apt update
sudo apt install build-essential cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## Create an untrained model

```bash
./build/simann init --model sima.snn --hidden 96
./build/simann info --model sima.snn
```

## Train

Use UTF-8 plain text in a corpus file. A few kilobytes are enough to smoke-test the pipeline, but meaningful language modeling requires much more diverse text. Use material you have permission to train on.

```bash
./build/simann train --data dataset.txt --model sima.snn --steps 10000 --seq 64 --lr 0.001
```

The trainer uses the first 90% for optimization and the final 10% for validation. It reports training and validation cross-entropy loss. Every 1,000 steps and at the end, it saves a checkpoint. Running the same command again resumes from that checkpoint.

## Generate text

```bash
./build/simann generate --model sima.snn --prompt "سلام" --tokens 300 --temperature 0.85 --top-k 32
```

## Hardware notes

For an Intel Pentium Dual-Core E2160, 4 GB RAM and no CUDA GPU:

- Start with `--hidden 64` or `96`.
- Start with `--seq 32` or `64`.
- Use Release mode and `-j2`.
- Training time depends on corpus size and CPU clock; no fixed duration is implied.
- Do not run multiple training jobs simultaneously.

## Known limitations

- This is a compact byte-level RNN, not a Transformer.
- It performs next-byte continuation, not instruction tuning or dialogue alignment.
- There is no pretrained knowledge, tokenizer vocabulary, dataset, or model weights in this repository.
- Byte-level tokenization is universal but inefficient for natural language; a subword tokenizer and a gated recurrent/Transformer architecture are future upgrades.
- Checkpoints are currently intended for builds using the same compiler/platform/struct layout.

