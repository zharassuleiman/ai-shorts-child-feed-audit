# Prevalence and Algorithmic Amplification of AI-Generated Child-Directed Shorts on YouTube

**Authors:** Zharas Suleiman
**Course:** Research Methods
**Methodology card:** [docs/methodology_card.md](docs/methodology_card.md)

## Research Question

How much AI-generated content do YouTube Shorts feeds served to child-profile accounts contain, do watch-time signals amplify it, and can it be detected reliably by multimodal classifiers?

| ID | Question | Test |
|----|----------|------|
| RQ1 | Is the AI-generated share higher in child-profile feeds than in adult-profile feeds? | One-sided two-proportion z-test |
| RQ2 | Does dwelling on AI-generated videos increase the AI share of later recommendations more than dwelling on human-made videos? | One-sided Welch t-test on share change |
| RQ3 | Does a multimodal detector beat a metadata-only baseline by at least 0.05 macro-F1? | Held-out macro-F1 comparison |

## Overview

A sock-puppet audit: automated child and adult profiles collect public YouTube Shorts recommendations under a fixed session protocol and three watch-policy arms (AI-dwell, human-dwell, no-dwell). Collected videos are annotated with a codebook (Cohen's kappa >= 0.70) and used to train and evaluate AI-generation detectors. **No real children and no personal data are involved; only public content is analysed and results are reported in aggregate.**

> The files in `data/sample/` are **synthetic** micro-samples used only to verify that the pipeline runs. They are not real measurements.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── Dockerfile
├── requirements.txt          # exact version pins
├── configs/config.yaml       # all experiment parameters
├── data/sample/              # synthetic 8-10 row samples (RQ1, RQ2, RQ3)
├── src/benchmark.py          # measurement script
├── notebooks/                # exploratory analysis
├── docs/methodology_card.md  # Methodology Passport Card
└── results/                  # output (git-ignored)
```

## Planned Benchmarks

| Role | Metric | RQ |
|------|--------|----|
| Primary | Share of AI-generated videos among first 100 shorts/session | RQ1 |
| Primary | Change in AI share, baseline -> session 5 | RQ2 |
| Primary | Macro-F1 on held-out test set | RQ3 |
| Guardrail | Cohen's kappa >= 0.70 on double-coded subset | all |
| Guardrail | Min. sessions / accounts per group (config) | RQ1, RQ2 |
| Guardrail | Run time < `timeouts.run_seconds` | all |
| Baseline | Adult profiles / human-dwell arm / metadata-only classifier | RQ1 / RQ2 / RQ3 |

## System Requirements

Docker >= 24, **or** Python 3.12 with `pip install -r requirements.txt`.

## Quickstart

```bash
git clone https://github.com/<zharas07>/ai-shorts-child-feed-audit.git && cd ai-shorts-child-feed-audit
docker build -t shorts-audit .
docker run --rm shorts-audit
```

Without Docker: `pip install -r requirements.txt && python src/benchmark.py`.
The run prints a JSON result and writes `results/result.json` (inside the container, to see it on the host add `-v "$(pwd)/results:/app/results"`).

## License and Citation

Released under the MIT License. If you use this work, cite:

```
Suleiman, Z. (2026). Prevalence and Algorithmic Amplification of AI-Generated
Child-Directed Shorts on YouTube: A Sock-Puppet Audit. GitHub repository.
```
