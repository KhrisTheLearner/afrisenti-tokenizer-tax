# Security

## Reporting a problem

If you find a security problem in this repository, such as a leaked credential, personal data that should not be here, or code that could harm someone who runs it, please report it privately: open the **Security** tab of this repository and choose **Report a vulnerability**. Please do not open a public issue for it. I will answer as soon as I can.

## What the code does and does not do

- It downloads the AfriSenti dataset and two public models, AfroXLMR-base and LaBSE, from the Hugging Face Hub. Cells 1.1 and 1.3 record the commits they downloaded and stop if Kinyarwanda no longer reproduces the earlier counts, and the grid script loads the models at those recorded commits.
- It never sets `trust_remote_code`, so no code from the Hub runs on your machine.
- It does not call any other web service and does not run shell commands.
- The grid script saves the Kinyarwanda and Swahili models in `models/` with joblib and Transformers. Pickle-based files like these can run code when they are loaded, so load only models you trained yourself with the grid script.

## Pinned versions with published advisories

`requirements.txt` pins the exact versions the results were produced with, so that the numbers in the paper can be reproduced. Some of those versions now have published advisories (checked with pip-audit in October 2026):

| Package | Pinned | Advisories | Why they do not reach this code |
|---|---|---|---|
| nltk | 3.9.4 | 54, most fixed by 3.10.3 | Only `TweetTokenizer` is used, on the pinned dataset. No NLTK data is downloaded or loaded. |
| transformers | 4.57.6 | 9, fixed in 5.x | Only the two pinned models are loaded, without remote code, and training never resumes from an outside checkpoint. |
| datasets | 5.0.0 | 1, fixed in 5.0.1 | The advisory concerns folder-based dataset builders; this code reads parquet files at a recorded commit. |
| torch | 2.12.1 | 1, fixed in 2.13.0 | The advisory concerns `torch.jit.script`, which this code does not use. |

Run the code in its own virtual environment, as the README shows. If you reuse it for new work rather than to reproduce the paper, upgrade these packages first; a newer NLTK may count `TweetTokenizer` words slightly differently.
