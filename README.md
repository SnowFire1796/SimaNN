# SimaNN

**SimaNN** is a from-scratch, CPU-first C++26 neural language-model project. It starts with random weights and is deliberately **untrained**. You supply a plain-text corpus and train it locally.

## Model

The first implementation is a byte-level Elman RNN language model:
- 256-token vocabulary (all byte values; UTF-8 corpora are accepted as byte streams)
- 96 hidden units by default
- tanh recurrent state and softmax next-byte prediction
- truncated backpropagation through time
- Adam optimizer, global gradient clipping
- top-k and temperature sampling
- binary checkpoints including optimizer state

Default parameter count: 59,? (printed by `simann info`; exact count depends on hidden size).

This intentionally favors a small, understandable model that can run on a dual-core Pentium and 4 GB RAM. It is not comparable to a pretrained modern LLM. Byte-level generation can occasionally emit invalid UTF-8 sequences, especially before sufficient training.

## Build (Xubuntu)

```bash
sudo apt update
sudo apt install build-essential cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## Initialize an untrained model

```bash
./build/simann init --model sima.snn --hidden 96
./build/simann info --model sima.snn
```

## Train on your own text

Put UTF-8 text in a file, one or more documents concatenated together. Start with a clean corpus containing material you have permission to use.

```bash
./build/simann train --data dataset.txt --model sima.snn --steps 10000 --seq 64 --lr 0.001
```

Training reads fixed-size windows from the file instead of loading the entire corpus into RAM. Re-running the same command resumes the checkpoint when it exists. Backups of checkpoints are recommended.

## Generate

```bash
./build/simann generate --model sima.snn --prompt "سلام" --tokens 300 --temperature 0.85 --top-k 32
```

## Hardware guidance

For the Pentium Dual-Core E2160 / 4 GB RAM system:
- Start with `--hidden 64` or `96`.
- Use `--seq 32` or `64`.
- Build with Release mode and `-j2`.
- Training time depends heavily on corpus size and CPU clock; no fixed duration is promised.
- Avoid running multiple training processes simultaneously.

## Limitations / roadmap

This is a compact educational language model, not a transformer or a conversational assistant out of the box. It learns byte-level continuation from the corpus; it does not inherently learn instruction following, factual grounding, dialogue roles, or semantic tokenization. The current trainer reports training loss; a true held-out validation evaluator, tokenizer improvements, multi-layer GRU/LSTM/Transformer blocks, and richer sampling controls are planned.

No pretrained weights or datasets are included.
