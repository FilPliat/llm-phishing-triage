# LLM-Assisted Phishing Email Triage for SOC Environments

Source code and the UC2 dataset for the MSc thesis of **Filippos Pliatsikas**, Conservatoire national des arts et métiers (Cnam), École Cnam Grèce, 2026.

The thesis builds a phishing triage tool around GPT-4o, used through the OpenAI API without additional training. It compares the tool with traditional machine-learning baselines, a fine-tuned BERT model and reputation-based rules on two use cases:

- **UC1:** standard phishing, from public email datasets.
- **UC2:** trusted-source phishing sent from compromised partner accounts and hosted on trusted cloud platforms. This is a synthetic dataset built for the thesis.

## Contents

| Path | What it is |
|---|---|
| `01_…` to `13_…` | Numbered scripts, run in order. Each one produces specific tables or figures of the thesis. |
| `05_bert_uc1_colab.ipynb` | BERT fine-tuning notebook (Google Colab, GPU). |
| `src/` | Shared code: data loading, the GPT-4o classifier and prompts, baselines, evaluation. |
| `config.py` | Folder locations and shared settings. |
| `data/processed/uc2_dataset.csv` | The UC2 dataset used for all UC2 results (3,000 emails). |
| `HOW_TO_RUN.txt` | Full setup and step-by-step instructions. |
| `.env.example` | Template for the OpenAI API key. |

## Quick start

1. Install Python 3.10 or newer, then run `pip install -r requirements.txt`.
2. For the GPT-4o steps, copy `.env.example` to `.env` and put your own OpenAI API key in it. Never upload `.env`.
3. For the UC1 steps (01–05), download the public datasets and place them in the `data` folder. The expected file names are listed in `HOW_TO_RUN.txt`. The UC1 datasets are not redistributed here.
4. Run the scripts in numeric order, as described in `HOW_TO_RUN.txt`.

The steps that call the OpenAI API (04, 06, 07, 08 and 09) have an API cost. All other steps run locally.

## Notes

- GPT-4o is called with the snapshot `gpt-4o-2024-08-06` at temperature 0. A new run can still differ slightly from the thesis results.
- The prompt wording in `src/llm_classifier.py` was lightly edited for readability after the experiments, without changing its meaning.
- The UC2 emails were generated with GPT-4o at temperature 0.9. Running step 06 creates a new dataset whose emails will not be identical to the included one.
- "Thesis Codes" in the scripts and in `HOW_TO_RUN.txt` means the main folder of this repository.

## License

MIT License. See `LICENSE`.
