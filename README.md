# Emotion Detection on Financial News

A Human Language Technologies (HLT) course project for the 2023/24 academic year, investigating sentiment in financial news through traditional machine learning, neural networks, and large language models.

The experiments classify foreign exchange news as **Positive**, **Negative**, or **Neutral**. Although the project uses the broader term *emotion detection*, its supervised prediction task is three-class sentiment classification. Emotional dimensions are explored separately through affective lexicons. A possible future application is sentiment-aware news recommendation.

## Project approach

The project compares models with different representations of financial language:

- A custom **Naive Bayes** classifier uses headline word occurrences and the currency-pair ticker, with Laplace smoothing for words.
- A **Conditional Random Field (CRF)** uses token features, neighboring words, part-of-speech tags, and lexicon scores. Each token receives its headline's sentiment as its training label.
- A **Multi-Layer Perceptron (MLP)** uses sentence vectors from `sent2vec` and one hidden layer with 80 neurons and ReLU activation.
- A custom **Transformer encoder**, implemented in TensorFlow/Keras, processes article text using a CodeLlama tokenizer, positional embeddings, self-attention, and a classification head.
- **Llama 3 8B Instruct** and **Phi 3 Mini 4K Instruct** classify lowercased headlines through prompting. Their notebooks compare zero-shot predictions with supervised fine-tuning using 4-bit model loading and LoRA adapters through Unsloth.
- Additional **FinBERT** experiments use `ProsusAI/finbert` for pretrained inference and fine-tuning.

The Llama and Phi experiments each include versions with and without lexicon scores in the prompt, testing whether explicit affective information improves predictions.

## Dataset

The included `sentiment_annotated_with_texts.csv` is identified in the project materials as the **Forex News Annotated Dataset**, with its source recorded on [Zenodo](https://zenodo.org/records/7976208).

The copy in `Final code/` contains **2,291 records**, covering January 12–May 4, 2023, from **FX Street** and **Forex Live**. It covers five currency pairs: `AUDUSD`, `EURCHF`, `EURUSD`, `GBPUSD`, and `USDJPY`. There are 2,265 distinct title strings, so the record count should not be interpreted as the number of unique headlines.

| Sentiment label | Records |
| --- | ---: |
| Neutral | 815 |
| Positive | 767 |
| Negative | 709 |
| **Total** | **2,291** |

| Columns | Content |
| --- | --- |
| `published_at`, `ticker` | Publication timestamp and currency pair |
| `true_sentiment` | Reference sentiment label used for supervision |
| `title`, `text` | News headline and full article text |
| `author`, `url`, `source` | Article metadata |
| `finbert_sentiment`, `finbert_sent_score` | FinBERT sentiment and score supplied with the data |

Most classifiers use `title` as input and `true_sentiment` as the target. The custom Transformer notebook uses `text`. Preprocessing differs by experiment: the introductory analysis and Naive Bayes notebooks clean punctuation and lemmatize text, while the Llama and Phi notebooks primarily lowercase headlines.

## Affective lexicons

Two lexicon files are included in both `Final code/` and `Project files/`:

- **NRC-VAD** (`NRC-VAD-Lexicon.txt`) provides valence, arousal, and dominance scores, representing pleasantness, activation, and control.
- **SentiWordNet 3.0** (`SentiWordNet_3.0.0.txt`) provides positive and negative sentiment scores.

The CRF uses these scores as token features. In the Llama and Phi variants with lexicons, matched scores are aggregated for each headline and inserted into the classification prompt. The introductory notebook also explores TextBlob polarity and visualizes affective scores.

## Repository layout

```text
.
├── Final code/                 Final experiment notebooks, data, and lexicons
│   ├── Introduction to data.ipynb
│   ├── Naive Bayes.ipynb
│   ├── CRF.ipynb
│   ├── MLP.ipynb
│   ├── Transformer.ipynb
│   ├── LLama-3-8b-With-Lexicons.ipynb
│   ├── LLama-3-8b-Without-Lexicon.ipynb
│   ├── Phi-3-With-Lexicons.ipynb
│   ├── Phi-3-Without-Lexicons.ipynb
│   └── requirements_*.txt      Recorded environments for individual experiments
├── Project files/              Earlier experiments, spreadsheets, and data copies
├── Models/finbert_model/       FinBERT model configuration
├── Papers/                    Background research PDFs
├── Presentations/             Proposal, final slides, and presentation notes
├── FinBERT.ipynb               Additional FinBERT experiments
└── LLama-3-8b-With-Lexicons.ipynb  Additional Llama experiment version
```

Start with `Final code/Introduction to data.ipynb` for data exploration, then choose a model notebook from `Final code/`. Each model notebook loads its own inputs; running all preceding experiments is unnecessary. `Project files/` preserves development versions and alternative approaches.

Model weights with the `.safetensors` extension are excluded by `.gitignore`. A checkout may therefore contain the FinBERT configuration without its weights. The Llama and Phi notebooks load base models from Hugging Face or local model directories and write checkpoints or exported models when their training and save cells are executed.

## Running the notebooks

These are research notebooks originally used in Google Colab and local GPU environments. Their paths and some setup cells need adjustment before execution.

### Start with data exploration or Naive Bayes

The following is a starting environment based on the core versions recorded in the repository; it is not a verified installation of every experiment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install notebook "numpy==1.25.2" "pandas==2.0.3" \
  "scipy==1.11.4" "scikit-learn==1.2.2" "spacy==3.7.4" \
  "textblob==0.17.1" matplotlib seaborn
python -m spacy download en_core_web_sm
cd "Final code"
python -m notebook
```

Open `Introduction to data.ipynb` or `Naive Bayes.ipynb`. Replace their Google Drive input paths with paths relative to the notebook directory, for example:

```python
news_df = pd.read_csv("sentiment_annotated_with_texts.csv")
```

For lexicon-loading cells, use `NRC-VAD-Lexicon.txt` and `SentiWordNet_3.0.0.txt` in the same directory. Skip Google Drive mounting cells when running locally, and update any output paths before executing export cells.

On Colab, mount Drive and either reproduce the directory structure expected by the notebook or update its paths to point to your uploaded files.

### Model-specific environments

| Experiment | Environment reference | Additional components |
| --- | --- | --- |
| Data exploration | `requirements_introduction.txt` | spaCy English model, TextBlob, plotting libraries |
| Naive Bayes | `requirements_Naive_Bayes.txt` | spaCy English model, scikit-learn |
| CRF | `requirements_CRF.txt` | `sklearn-crfsuite`, `python-crfsuite`, NLTK resources downloaded by the notebook |
| MLP | `requirements_MLP.txt` | `sent2vec`, PyTorch/Transformers for embeddings, TensorFlow/Keras imports |
| Custom Transformer | `requirements_Transformer.txt` | TensorFlow 2.16.1, Keras 3.3.3, TensorFlow Text, Hugging Face tokenizer |
| Llama 3 / Phi 3 | `requirements_Phi-3-and-Llama-3.txt` | CUDA GPU, PyTorch, Unsloth, bitsandbytes, TRL, PEFT, Accelerate, attention libraries |

All environment reference files above are in `Final code/`. They are full environment exports containing unrelated packages, Colab-specific wheels, and local `file://` dependencies. Use them to identify the original versions and build an environment for the chosen experiment; installing them unchanged on another machine may fail. Keep the TensorFlow and LLM environments separate where their dependencies differ.

The recorded LLM environment includes PyTorch `2.3.0+cu121`, Flash Attention `2.5.8`, and pinned Git revisions of Transformers and Unsloth. The notebook installation cells also contain unpinned upgrades, so they do not necessarily recreate that environment. The presentation notes describe fine-tuning on a GPU with 12 GB of memory; actual requirements depend on model and training settings.

Before running the LLM notebooks:

1. Update dataset, lexicon, checkpoint, and model-directory paths.
2. Prepare compatible GPU dependencies and access to the selected Hugging Face model.
3. Skip early memory-cleanup cells that delete `model`, `trainer`, or other variables before they exist in a fresh kernel.
4. In `Phi-3-With-Lexicons.ipynb`, correct the unnamed `= 0.05` argument in `FastLanguageModel.get_peft_model` to `lora_dropout = 0.05`.
5. Run the experiment cells in order, inspecting zero-shot results before training and evaluating the adapted model afterward.

The variants without lexicons still load lexicon files and compute scores in helper functions; their prompts omit those scores, so the files remain required by the current implementation.

## Results saved in the notebooks

The following values come from stored outputs in `Final code/`. They have not been reproduced as part of this documentation.

| Model | Zero-shot accuracy | Fine-tuned accuracy | Test records |
| --- | ---: | ---: | ---: |
| Llama 3 8B with lexicons | 63.4% | **85.4%** | 459 |
| Llama 3 8B without lexicons | 59.5% | 77.6% | 459 |
| Phi 3 Mini with lexicons | 64.5% | 81.7% | 459 |
| Phi 3 Mini without lexicons | 67.1% | 81.9% | 459 |

| Other model | Saved accuracy | Evaluation unit |
| --- | ---: | --- |
| Naive Bayes | 67% | 455 headlines after preprocessing, 80/20 split |
| CRF | 70% | 5,144 tokens from held-out headlines |
| MLP | 59% | 459 headlines, stratified 80/20 split |

The highest saved headline-classification accuracy among these final experiments is **85.4% for fine-tuned Llama 3 8B with lexicons**. Lexicon prompts improve the saved Llama results, while the saved fine-tuned Phi results are close, with a slightly higher accuracy without lexicons.

These results use different preprocessing, splits, and evaluation units. In particular, the CRF report flattens token predictions and does not measure one prediction per headline. Some shuffling is unseeded, so equal test-set sizes do not establish that runs use the same examples. Treat the tables as summaries of individual experiments rather than a controlled benchmark. Naive Bayes also includes five-fold evaluation; the custom Transformer and additional FinBERT experiments have their own outputs outside these comparisons.

## Supporting materials

`Presentations/final exam pp.pptx` and `Presentations/MANUS final pp.docx` explain the motivation, model choices, and conclusions. `Presentations/project_proposal.pptx` records the proposal and intermediate progress. The PDFs in `Papers/` provide background on emotion classification and news datasets.

The project materials identify expansion of the dataset and integration with news recommendation systems as directions for future work.
