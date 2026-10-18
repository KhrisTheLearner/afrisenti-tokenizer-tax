# Tokenizer tax and downstream performance in twelve African languages

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23443962.svg)](https://doi.org/10.5281/zenodo.23443962)

A subword tokenizer built mostly on English cuts the words of many African languages into more pieces than English words. People call this the token tax. In a preliminary study on Kinyarwanda, my first language, fine-tuning a pretrained model beat a simple TF-IDF baseline by six macro F1 points. This repository asks whether that holds across languages: does the number of subword tokens a vocabulary spends on a word predict how much a pretrained model gains over TF-IDF?

I ran the same three sentiment pipelines, with identical settings and seeds, on the twelve [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) languages that have training splits: TF-IDF with logistic regression, frozen LaBSE embeddings with the same classifier, and AfroXLMR-base fine-tuned with three seeds. That makes 60 runs. The test of the main question was fixed before the runs.

The paper is in [`paper/`](paper/Mumaragishyika_2026_tokenizer_tax.pdf). The code, results and suite are archived on Zenodo at [10.5281/zenodo.23443962](https://doi.org/10.5281/zenodo.23443962), which always points to the latest version; version v1.0.0 is [10.5281/zenodo.23443963](https://doi.org/10.5281/zenodo.23443963).

## What I found

- Fine-tuning beats TF-IDF in four languages, loses in five and ties in three.
- The pre-specified test, Spearman's ρ between subword tokens per word and the fine-tuning gain, is 0.00, with a 95 percent interval from −0.72 to 0.69. It stays near zero under four word definitions and with training size held fixed.
- Pretraining coverage and training size separate the languages better. The two languages that no pretraining list includes, Xitsonga and Twi, lose the most.
- The released splits carry artifacts a model can see, among them lower-cased Nigerian test sets and placeholders present in one split only. Adding the `@user` placeholder to the Swahili test set costs the two encoders 8.4 and 8.6 macro F1 points.
- On a 66-item adversarial suite in Kinyarwanda and Swahili, no pipeline reads a polite formal complaint as negative, and in Kinyarwanda only the encoders read negation.

## What each file does

| File | What it does | What it writes |
|---|---|---|
| `notebooks/01_data.ipynb`, Cell 1.1 | Loads the twelve languages from the Hugging Face parquet conversion of AfriSenti at a pinned commit, normalizes the tweets (runs of mentions collapsed, links turned into `http`, spaces cleaned, and a pair of straight quotes wrapping a whole tweet removed, which touches the 1,999 Amharic test tweets only), and removes training rows that copy a validation or test tweet | `data/afrisenti_audit.csv`, and `data/afrisenti_normalized.parquet` (not stored here) |
| Cell 1.2 | Retypes the 1,026 Kinyarwanda test tweets with texting rules, seed 42 | `data/kin_test_retyped.csv` (not stored here) |
| Cell 1.3 | Tokens per word under the AfroXLMR and LaBSE vocabularies, unknown tokens and the out-of-vocabulary rate | `data/afrisenti_fertility.csv`, `data/kin_retyped_tokens.csv` |
| Cell 1.4 | The pretraining coverage class of each language, read from the models' papers | `data/coverage.csv` |
| `scripts/run_grid.py` | The three pipelines on a list of languages; resumable, one row per finished run | `results/grid.csv`, `results/predictions/`, the LaBSE embeddings in `data/labse_*.npz` (not stored here, about 300 MB) and the Kinyarwanda and Swahili models in `models/` (not stored here, about 7 GB) |
| `notebooks/02_results.ipynb`, Cell 2.1 | The retyping test | `results/table_retyping.csv` |
| Cell 2.2 | Results per language with paired bootstrap intervals, and the compute table | `results/table_results.csv`, `results/table_runs.csv`, `results/table_compute.csv` |
| Cell 2.3 | The correlation tests and the main figure | `results/correlations.csv`, `results/gain_by_coverage.csv`, `results/fig_gain_vs_tpw.pdf` |
| Cell 2.4 | Near-copies between training and test | `results/near_copies.csv`, and `results/near_copies_test_tweets.csv` (not stored here) |
| Cell 2.5 | Letter case, script and placeholders in each split | `results/case_audit.csv`, `results/script_audit.csv`, `results/placeholder_audit.csv` |
| Cell 2.6 | Perturbation tests on the kept models | `results/table_perturbations.csv` |
| Cells 2.7a and 2.7 | The adversarial suite, written as rows and checked | `data/suite_items.csv` |
| Cell 2.8 | The suite scored by the kept models | `results/suite_predictions.csv`, `results/table_suite.csv`, `results/table_suite_by_label.csv` |
| Cell 2.9 | The paper's tables as LaTeX | `results/tables/` |
| `paper/` | The paper and its LaTeX source (ACL format) | |

Every result is written to a file the moment it exists, and every Markdown note under a cell was written after that cell's real output.

## Run it

Python 3.12. The grid and Cells 2.6 and 2.8 need a GPU; my runs used one NVIDIA RTX 4000 Ada. Everything else runs on a CPU.

```bash
git clone https://github.com/KhrisTheLearner/afrisenti-tokenizer-tax.git
cd afrisenti-tokenizer-tax
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Open `notebooks/01_data.ipynb` and run all cells. Cell 1.1 downloads the dataset from the Hugging Face Hub, records the commit it got in `data/afrisenti_audit.csv`, and stops if the split sizes differ from the ones in the paper. Cell 1.3 records the two tokenizers' commits in `data/afrisenti_fertility.csv` and stops if Kinyarwanda no longer reproduces the preliminary study's tokenizer counts, so a changed file upstream cannot slip in unnoticed.
2. Run the grid from the repository folder. It loads the two models at the commits Cell 1.3 recorded. Kinyarwanda goes first, and the script stops if it does not reproduce the preliminary study. The 60 runs took about two hours on a shared server, and a run that is already finished is skipped.

   ```bash
   nohup python scripts/run_grid.py kin swa tso arq por twi pcm ary amh yor ibo hau > results/grid.log 2>&1 &
   ```

3. Open `notebooks/02_results.ipynb` and run all cells. Cells 2.6 and 2.8 load the kept models from `models/`.

## Settings and seeds

- TF-IDF: word unigrams and bigrams plus character n-grams of length 2 to 5 inside word boundaries, minimum document frequency 2, sublinear term frequency, and a token pattern that keeps apostrophes and hyphens inside a word. Logistic regression with C = 1, balanced class weights, 2,000 iterations, random_state 42.
- LaBSE: `sentence-transformers/LaBSE`, batch size 64, the same logistic regression on the embeddings.
- AfroXLMR: `Davlan/afro-xlmr-base`, three epochs, learning rate 2e-5, batch size 32 (64 for evaluation), maximum length 128, bfloat16, the epoch with the best validation macro F1 kept, seeds 42, 43 and 44.
- Paired bootstrap intervals: 1,000 resamples per language, seed 42. Correlation tests: 10,000 permutations and 10,000 bootstrap resamples of the languages, seed 42.

## Versions

Python 3.12.9, numpy 2.3.5, pandas 3.0.3, pyarrow 24.0.0, scipy 1.16.1, scikit-learn 1.8.0, torch 2.12.1, transformers 4.57.6, sentence-transformers 5.6.0, datasets 5.0.0, huggingface_hub 0.36.2, nltk 3.9.4.

- AfroXLMR-base at commit `25f27299c247a6b73a767bd82d12444138b19337`
- LaBSE at commit `836121a0533e5664b21c7aacc5d22951f2b8b25b`
- AfriSenti parquet conversion at commit `676cfb15a6761c643d860dcb53bac976e5db2dfc`

## Data

AfriSenti (Muhammad et al., EMNLP 2023), [`shmuhammad/AfriSenti-twitter-sentiment`](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) on the Hugging Face Hub, CC BY 4.0, the official splits of the twelve languages with training data. The tweets were anonymized by the dataset's authors.

This repository stores no tweets. The files that hold them are rebuilt by the notebooks from the pinned dataset. A few outputs in the notebooks printed example tweets; in this copy those lines are replaced by a short note, and running the cell shows them again.

## The adversarial suite

`data/suite_items.csv` holds 66 items in Kinyarwanda and Swahili, each with a label, an English gloss and a note. The Kinyarwanda items are mine; I am a native speaker. I drafted the Swahili items as a non-native speaker, and two native speakers of Tanzanian Swahili, who asked to remain anonymous, revised them. My drafts are kept in the note column. The items and labels were committed before any model scored them.

## Notes on reading the notebooks

- In the comments, e101 to e501 are my notebooks from the preliminary study. Cells 1.1 and 1.3 stop unless Kinyarwanda reproduces the counts that study printed, so the twelve languages go through the same operations.
- The printed outputs come from the October 2026 runs. The project folder was then called `final_project`, which is the name the printed paths show. The only code change since those runs is the folder check at the top of each notebook cell: it looked for that name and now looks for the `data/` folder. Two comments in the grid script were updated the same way.
- The suite was committed in my earlier repository (commit `9f721ac`, the one the paper cites) before any model scored it. This repository starts from the final state, so that commit is not in its history.

## What this does not claim

- Twelve languages are few for a correlation. The interval around ρ is wide, so the result is no detectable effect, not proof that fertility never matters.
- This is one task (tweet sentiment), one dataset and two vocabularies.
- For Algerian Arabic, Moroccan Arabic and Mozambican Portuguese, the pretraining lists name the standard language, not the variety in the tweets.

## Security

[SECURITY.md](SECURITY.md) explains how to report a problem privately and which pinned versions now have published advisories.

## License

The code is under the MIT License. The suite and the tables are under CC BY 4.0. AfriSenti keeps its own CC BY 4.0 license; please cite it if you use the data.

## Citation

Please cite the paper:

```bibtex
@misc{mumaragishyika2026tokenizertax,
  title  = {Tokenizer Tax and Downstream Performance in Twelve African Languages},
  author = {Mumaragishyika, Christophe},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/KhrisTheLearner/afrisenti-tokenizer-tax}
}
```

To cite the exact code and results behind the paper, cite the archived release:

```bibtex
@software{mumaragishyika2026tokenizertaxcode,
  author    = {Mumaragishyika, Christophe},
  title     = {Tokenizer Tax and Downstream Performance in Twelve African Languages: code and results},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23443963},
  url       = {https://doi.org/10.5281/zenodo.23443963}
}
```

## Acknowledgments

This study grew out of a course project at the University of Missouri. I used Claude (Anthropic) for planning, code drafting, debugging and writing help; I ran all the code myself and checked every claim against the printed output.
