# Methodology Passport Card

## 1. Project Information
- **Title:** Prevalence and Algorithmic Amplification of AI-Generated Child-Directed Short Videos on YouTube Shorts: a Sock-Puppet Audit with Multimodal Detection
- **Author:** Zharas Suleiman
- **GitHub repository:** https://github.com/<zharas07>/ai-shorts-child-feed-audit

## 2. Research Type
**Quantitative.** All three research questions reduce to numeric outcomes (share of AI-generated content, change in that share, macro-F1) and are tested with statistical tests at a fixed significance level. The only qualitative element is the annotation codebook, which supports labelling and is not a standalone analysis method.

## 3. Hypotheses (from Task 1, refined)
| RQ | H0 | H1 |
|----|----|----|
| RQ1 | p_child = p_adult | p_child > p_adult (one-sided, α = 0.05) |
| RQ2 | Mean change in AI share after 5 sessions is equal across watch policies | Change is larger in the AI-dwell arm than in the human-dwell arm |
| RQ3 | macro-F1 (multimodal) ≤ macro-F1 (metadata-only) | macro-F1 (multimodal) exceeds the baseline by at least 0.05 |

## 4. Variable Matrix
| RQ | Independent variables | Dependent variables | Controlled variables |
|----|-----------------------|---------------------|----------------------|
| RQ1 | Profile type: child vs adult (declared age, kids-interest seeds) | Proportion of AI-generated videos among the first 100 shorts per session | Platform (YouTube Shorts), region and language, device/browser, time window, account age, logged-in state, session length, seed topics |
| RQ2 | Watch policy: AI-dwell / human-dwell / no-dwell | Change in AI share from baseline session to session 5 | Same seed topics per arm, equal total watch time, fixed number of videos per session, video length band, time of day, account cohort size |
| RQ3 | Feature set: metadata-only / visual / audio / visual + audio + metadata | Macro-F1 on held-out test set (plus precision and recall) | Fixed stratified train/test split and random seed (42), class balance, video duration band, equal hyperparameter-search budget, same annotators and codebook; library versions pinned in `requirements.txt` |

## 5. Metrics and Baseline
**Primary metrics**
- RQ1: share of AI-generated videos in the feed (p-value of the two-proportion z-test)
- RQ2: change in AI share (p-value of Welch's t-test)
- RQ3: macro-F1 on the held-out set (gain ≥ 0.05)

**Guardrail metrics**
- Cohen's κ ≥ 0.70 on the double-coded subset
- Minimum number of sessions/accounts per group (set in config)
- No personal data and no real children in the dataset
- Pipeline run time < 60 s on the test sample

**Baseline**
- RQ1: adult-profile accounts
- RQ2: human-dwell arm (equal watch time on human-made content); no-dwell as an additional control
- RQ3: metadata-only classifier
