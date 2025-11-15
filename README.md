# GPT-2 Fine-Tuning Pipeline (Hugging Face Transformers)

This repository contains a **modular pipeline** for fine-tuning OpenAI’s GPT-2 model on a custom text corpus using the Hugging Face `transformers` library.

The project is split into small, focused scripts:

- Install dependencies
- Load GPT-2 and tokenizer
- Create a tokenized dataset and collator
- Split train / test
- Configure a `Trainer`
- Train and evaluate
- Visualize training loss

---

## 1. Project goals

1. Fine-tune GPT-2 on any plain-text corpus.
2. Keep the training code **clean, modular, and reusable**.
3. Quickly inspect model quality via evaluation metrics and training-loss plots.

Typical use cases:

- Domain-adapted GPT-2 (e.g., finance/legal/medical corpus).
- Experimenting with different datasets / hyperparameters.
- Teaching / demonstrating Hugging Face `Trainer` workflows.

---

## 2. Repository structure

```text
.
├─ README.md
├─ configure_trainer.py        # build Trainer with training arguments
├─ create_dataset.py           # create TextDataset + DataCollator
├─ evaluate_model.py           # evaluate trained model
├─ import_modules.py           # convenience imports (Torch, HF, Matplotlib)
├─ install_libraries.py        # pip installs (transformers, torch, datasets)
├─ load_gpt2.py                # load GPT-2 tokenizer, config, model
├─ main.py                     # end-to-end orchestration script
├─ train_model.py              # trainer.train() wrapper
├─ train_test_split.py         # random train/test split for dataset
└─ visualize_performance.py    # plot training metrics using Matplotlib
