# Speaker Script: Phase 1 — Demographic Fairness Experiments on Export_and_Compiled.xlsx

---

## Opening (Context & Motivation)

So the goal of this phase is to investigate **demographic fairness in Table Union Search annotations**. We have a dataset of 60 human annotators who each answered 8 table-union-search questions — that gives us 480 annotation rows total. Each annotator also provided demographic information: their age, education level, English proficiency, and field of study.

The core question we're asking is: **Are certain demographic groups of annotators systematically disadvantaged?** In other words, when we look at who gets questions right and wrong, do we see patterns tied to demographics rather than individual skill?

We approach this using the **FairEM framework** from Shahbazi et al. (VLDB 2023), which was originally designed for entity matching fairness. We adapted all 10 of their confusion-matrix-based fairness metrics to work with TUS annotation data. The key idea is: for each demographic group, compute a metric, subtract the overall population metric, and check if that disparity exceeds a threshold of 0.1 — if it does, we flag it as **UNFAIR**.

Before going into experiments, let me clarify our classification of sensitive attributes:

- **Active sensitive attributes** are inherent, immutable personal characteristics — things like age, gender, native language. In our study, **LinguisticGroup** (Native vs Non-Native English speakers) and **AgeGroup** (Young 18-34 vs Older 35+) are active attributes.
- **Passive sensitive attributes** are acquired or circumstantial — things like education level, location, or field of study. In our study, **ExpertiseGroup** (STEM vs Non-STEM major) and **ExperienceGroup** (High Education — Master's/Doctoral vs Lower Education) are passive attributes.

This distinction matters because bias from active attributes is more concerning from a fairness standpoint — you can't change your native language or age, so any systematic disadvantage tied to these is inherently unfair. Bias from passive attributes, while still worth investigating, reflects choices annotators have made.

---

## Experiment 1: Demographic Group Distributions

**What we did:** Before running any fairness analysis, we need to understand who our annotators are. We mapped all 60 annotators into our four binary demographic partitions and looked at the group sizes.

**Why:** Imbalanced groups can affect the reliability of fairness metrics. If one group has only 5 people, any disparity we find might just be noise.

**Results:**

| Partition | Group | Annotators | Percentage |
|---|---|---|---|
| LinguisticGroup (Active) | Native | 49 | 81.7% |
| LinguisticGroup (Active) | Non-Native | 11 | 18.3% |
| AgeGroup (Active) | Older-35plus | 38 | 63.3% |
| AgeGroup (Active) | Young-18-34 | 22 | 36.7% |
| ExpertiseGroup (Passive) | STEM | 42 | 70.0% |
| ExpertiseGroup (Passive) | Non-STEM | 18 | 30.0% |
| ExperienceGroup (Passive) | Lower-Edu | 39 | 65.0% |
| ExperienceGroup (Passive) | High-Edu | 21 | 35.0% |

**Observation:** The most imbalanced partition is LinguisticGroup — we only have 11 Non-Native speakers versus 49 Native. This is important to keep in mind for all subsequent experiments. The Non-Native group is our smallest, which means any findings there need to be interpreted cautiously. For the passive attributes, ExpertiseGroup is fairly skewed too — 70% STEM — which reflects the Qualtrics recruitment pool.

---

## Experiment 2: Full 10-Metric FairEM Disparity Table

**What we did:** This is the core fairness assessment. For each of the 8 demographic groups (2 per partition), we computed all 10 FairEM metrics — Accuracy Parity (AP), Statistical Parity (SP), True Positive Rate (TPR), True Negative Rate (TNR), Positive Predictive Value (PPV), Negative Predictive Value (NPV), False Positive Rate (FPR), False Negative Rate (FNR), False Discovery Rate (FDR), and False Omission Rate (FOR). Then we computed the disparity: group metric minus overall metric.

**Why:** Using all 10 metrics rather than just accuracy gives us a comprehensive view. A group might look fair on accuracy but be unfair on TPR — meaning they specifically miss positive cases even though their overall rate looks OK.

**Thought process:** We use a one-sided fairness test, matching the original FairEM paper. For "higher-is-better" metrics (AP, SP, TPR, TNR, PPV, NPV), a group is UNFAIR only if their disparity is below -0.1 — meaning they're disadvantaged. For "lower-is-better" metrics (FPR, FNR, FDR, FOR), UNFAIR means disparity above +0.1.

**Results — Overall performance:** 67.3% accuracy, 58.8% TPR across all 480 annotations.

**Key findings by partition:**

**Active Attributes:**

- **LinguisticGroup:** Non-Native AP=0.636 (d=-0.037), TPR=0.500 (d=-0.088). The TPR disparity of -0.088 is borderline — just below the -0.1 UNFAIR threshold but the largest disparity in the entire table. Non-Native speakers miss 50% of true positives versus 60.7% for Native. This active attribute shows the strongest signal.
- **AgeGroup:** Very small disparities across all metrics. Older-35plus AP=0.668 (d=-0.005), Young AP=0.682 (d=+0.009). No fairness concerns here.

**Passive Attributes:**

- **ExpertiseGroup:** Almost identical performance. STEM AP=0.673 (d=-0.0003), Non-STEM AP=0.674 (d=+0.001). This passive attribute shows essentially zero bias.
- **ExperienceGroup:** Minimal differences. High-Edu AP=0.667 (d=-0.006), Lower-Edu AP=0.676 (d=+0.003).

**Bottom line:** Under the standard FairEM threshold of 0.1, **ALL groups are classified as FAIR across all 10 metrics.** No group breaches the threshold. However, the Non-Native group (an active attribute) shows the closest-to-threshold disparity, especially on TPR.

---

## Experiment 3: Behavioral Signal Profiles & Per-Annotator Accuracy

**What we did:** We looked at behavioral features — decision time, first click time, click count, confidence level, and a "DecisionTimeFract" metric (first click divided by total decision time, measuring hesitation). We also plotted per-annotator accuracy distributions as box plots with individual data points.

**Why:** Fairness isn't just about outcomes — it's about understanding *how* different groups approach the task. If Non-Native speakers take longer but are less accurate, that tells a different story than if they answer quickly and carelessly.

**Results:**

| Group (Type) | Avg Decision Time | Avg Clicks | Avg Confidence | DecisionTimeFract |
|---|---|---|---|---|
| Native (Active) | 62.3s | 2.15 | 77.1 | 0.664 |
| Non-Native (Active) | 75.5s | 1.55 | 76.2 | 0.706 |
| Older-35plus (Active) | 64.2s | 1.78 | 77.8 | 0.703 |
| Young-18-34 (Active) | 65.6s | 2.48 | 75.5 | 0.618 |
| STEM (Passive) | 73.9s | 1.88 | 78.3 | 0.684 |
| Non-STEM (Passive) | 43.3s | 2.41 | 73.8 | 0.644 |
| High-Edu (Passive) | 56.9s | 2.05 | 81.5 | 0.658 |
| Lower-Edu (Passive) | 68.9s | 2.03 | 74.5 | 0.679 |

**Observations:**
- Non-Native speakers (active) take longer on average (75.5s vs 62.3s) but click less (1.55 vs 2.15). They have a higher DecisionTimeFract (0.706 vs 0.664), meaning they commit to their first click relatively faster — they may be less likely to explore alternatives.
- Young annotators (active) click much more (2.48 vs 1.78) and have a lower DecisionTimeFract (0.618), suggesting more exploratory behavior.
- High-Edu annotators (passive) report the highest confidence (81.5) — this becomes important in the calibration experiment.
- The per-annotator accuracy box plots show considerable within-group variation — some annotators in every group score 100% while others score below 25%. This tells us individual differences are large relative to group differences.

---

## Experiment 4: Majority Vote Disagreement

**What we did:** For each question, we computed the majority vote across all annotators. Then we measured how often each demographic group disagrees with the majority, and compared this to their actual error rate.

**Why:** If a group consistently disagrees with the majority, it could mean two things: either they're systematically wrong, or they're right when others are wrong. Separating these cases reveals whether minority opinions in certain groups carry useful signal.

**Results:**

| Group (Type) | Disagree w/ Majority | Annotator Error Rate | Majority Error Rate |
|---|---|---|---|
| Native (Active) | 33.7% | 31.9% | 12.5% |
| Non-Native (Active) | 26.1% | 36.4% | 12.5% |
| STEM (Passive) | 31.6% | 32.7% | 12.5% |
| Non-STEM (Passive) | 34.0% | 32.6% | 12.5% |
| High-Edu (Passive) | 31.6% | 33.3% | 12.5% |
| Lower-Edu (Passive) | 32.7% | 32.4% | 12.5% |
| Older-35plus (Active) | 31.9% | 33.2% | 12.5% |
| Young-18-34 (Active) | 33.0% | 31.8% | 12.5% |

**Observations:**
- Non-Native speakers actually disagree with the majority *less* (26.1%) than Native speakers (33.7%). But they have a *higher* error rate (36.4% vs 31.9%). This is an interesting pattern — Non-Native speakers tend to follow the crowd, but when the crowd is wrong on questions they find difficult, they get swept along. They're less likely to hold a contrarian correct answer.
- Majority error rate is uniformly 12.5% across all groups (1 out of 8 questions), which confirms the majority vote is generally reliable regardless of which subgroup we condition on.

---

## Experiment 5: Calibration Gap

**What we did:** For each group, we computed the "calibration gap" — the difference between mean self-reported confidence and actual accuracy. A positive gap means overconfidence.

**Why:** Overconfidence is a fairness concern because if we use confidence as a weighting signal (as many aggregation schemes do), overconfident groups would get disproportionate influence even when they're wrong.

**Results:**

| Group (Type) | Mean Confidence | Mean Accuracy | Calibration Gap | Flag |
|---|---|---|---|---|
| Native (Active) | 77.1% | 68.1% | +0.090 | fair |
| **Non-Native (Active)** | **76.2%** | **63.6%** | **+0.126** | **CALIB-UNFAIR** |
| Non-STEM (Passive) | 73.8% | 67.4% | +0.064 | fair |
| **STEM (Passive)** | **78.3%** | **67.3%** | **+0.110** | **CALIB-UNFAIR** |
| **High-Edu (Passive)** | **81.5%** | **66.7%** | **+0.148** | **CALIB-UNFAIR** |
| Lower-Edu (Passive) | 74.5% | 67.6% | +0.069 | fair |
| **Older-35plus (Active)** | **77.8%** | **66.8%** | **+0.110** | **CALIB-UNFAIR** |
| Young-18-34 (Active) | 75.5% | 68.2% | +0.073 | fair |

**Observations:**
- **Four groups exceed the 0.1 calibration gap threshold:** Non-Native (+0.126), STEM (+0.110), High-Edu (+0.148), and Older-35plus (+0.110).
- **High-Edu has the worst calibration** — they report the highest confidence (81.5%) but don't have the highest accuracy. This is a "Dunning-Kruger-adjacent" pattern: more education correlates with more confidence but not more accuracy on these specific TUS tasks.
- **Non-Native overconfidence** is particularly concerning because it's an active attribute. Non-Native speakers believe they're doing well (76.2% confidence) but are actually at 63.6% accuracy — a 12.6-point gap.
- This means any weighted voting scheme using confidence would amplify these groups' influence unfairly.

---

## Experiment 6: Explanation Rate

**What we did:** Our dataset includes whether each annotator provided a textual explanation for their answer (IsExp flag). We looked at explanation rates by group and whether providing an explanation correlates with accuracy.

**Why:** Explanation provision is a behavioral engagement signal. If certain groups explain less, it could indicate lower engagement or different cognitive strategies.

**Results:**

| Group (Type) | Explanation Rate | Acc with Explanation | Acc without |
|---|---|---|---|
| Native (Active) | 96.4% | 68.3% | 64.3% |
| Non-Native (Active) | 88.6% | 65.4% | 50.0% |
| Non-STEM (Passive) | 100.0% | 67.4% | N/A |
| STEM (Passive) | 92.9% | 68.0% | 58.3% |
| High-Edu (Passive) | 96.4% | 66.1% | 83.3% |
| Lower-Edu (Passive) | 94.2% | 68.7% | 50.0% |
| Older-35plus (Active) | 97.4% | 66.9% | 62.5% |
| Young-18-34 (Active) | 90.9% | 69.4% | 56.3% |

**Observations:**
- Non-Native speakers have the lowest explanation rate (88.6% vs 96.4% for Native). When they don't explain, their accuracy drops to 50% — essentially coin-flip performance. This suggests that when Non-Native speakers skip explanations, they may be guessing.
- Non-STEM annotators have 100% explanation rate — every single one provided explanations.
- The High-Edu group shows a counterintuitive pattern: those who *don't* explain actually score higher (83.3%). But this is based on very few observations (only 6 unexplained answers out of 168), so we shouldn't read too much into it.

---

## Experiment 7: Per-Question Accuracy Disparity

**What we did:** Instead of averaging across all 8 questions, we looked at each question individually. This reveals whether certain questions are harder for specific groups — a difficulty-by-demographic interaction.

**Why:** Aggregate fairness can hide question-level unfairness. A group might be fair overall but severely disadvantaged on one specific question type.

**Results — Overall question difficulty:**

| Question | Overall Accuracy | Difficulty |
|---|---|---|
| Q8 | 88.3% | Easy |
| Q4 | 75.0% | Easy |
| Q3 | 71.7% | Easy |
| Q1 | 70.0% | Easy |
| Q7 | 68.3% | Easy |
| Q2 | 58.3% | Hard |
| Q5 | 58.3% | Hard |
| Q6 | 48.3% | Hard |

**Critical finding — Q6 x Non-Native (Active attribute):**
- Non-Native speakers scored **9.1%** on Q6 (1 out of 11 correct) versus **57.1%** for Native speakers
- That's a disparity of **-0.392** — nearly four times the UNFAIR threshold
- This is the single largest disparity in the entire dataset by a wide margin

Other notable disparities:
- Non-Native on Q2: 45.5% vs 61.2% (d=-0.129, exceeds threshold)
- Non-Native on Q5: 72.7% vs 55.1% (d=+0.144, *better* than Native — pattern reversal!)
- High-Edu on Q7: 57.1% vs 68.3% (d=-0.112, exceeds threshold, passive attribute)
- Young-18-34 on Q2: 45.5% vs 58.3% (d=-0.129)

**Key insight:** Non-Native speakers (active attribute) actually outperform on some questions (Q5: +14.4%, Q8: +11.7%, Q1: +2.7%) but dramatically underperform on Q6 and Q2. This isn't uniform bias — it's *question-specific* bias concentrated in the hardest questions.

---

## Experiment 8: Conditional Fairness by Task Difficulty

**What we did:** Based on the Experiment 7 findings, we split all questions into Hard (Q2, Q5, Q6 — accuracy below 65%) and Easy (Q1, Q3, Q4, Q7, Q8) tiers. We then re-ran the FairEM disparity analysis within each tier separately.

**Why:** If bias is concentrated in hard questions, averaging across all questions dilutes the signal. Conditioning on difficulty reveals hidden unfairness that aggregate analysis misses.

**Results:**

| Tier | Group (Type) | Accuracy | Disparity | Flag |
|---|---|---|---|---|
| **Hard** | **Non-Native (Active)** | **42.4%** | **-0.126** | **UNFAIR** |
| Hard | Native (Active) | 57.8% | +0.028 | fair |
| Hard | All others | ~54-56% | < ±0.01 | fair |
| Easy | Non-Native (Active) | 76.4% | +0.017 | fair |
| Easy | Native (Active) | 74.3% | -0.004 | fair |

**This is the critical finding:** Non-Native speakers are the **only group flagged UNFAIR** in the entire study, and only on hard questions. On easy questions, they actually outperform Native speakers (76.4% vs 74.3%). The unfairness is entirely driven by the interaction between an active sensitive attribute (native language) and question difficulty.

**Fisher's exact test on Q6:** Native 28/49 correct vs Non-Native 1/11 correct. Odds ratio = 13.33, p-value = 0.006. This is statistically significant even with our small sample.

**None of the passive attributes (Expertise, Education) show any UNFAIR signal** in either the hard or easy tier.

---

## Experiment 9: Bootstrap Confidence Intervals

**What we did:** We ran 5,000 bootstrap resamples to compute 95% confidence intervals on all accuracy and TPR disparities. We also ran per-annotator Welch t-tests.

**Why:** With only 11 Non-Native speakers, we need to check whether the aggregate disparities are statistically meaningful or could arise from sampling noise.

**Results:**

| Group (Type) | Acc Disparity | 95% CI | Significant? |
|---|---|---|---|
| Non-Native (Active) | -0.037 | [-0.146, +0.071] | No |
| Native (Active) | +0.008 | [-0.054, +0.070] | No |
| STEM (Passive) | -0.0003 | [-0.069, +0.064] | No |
| Non-STEM (Passive) | +0.001 | [-0.086, +0.088] | No |
| High-Edu (Passive) | -0.006 | [-0.088, +0.074] | No |
| Lower-Edu (Passive) | +0.003 | [-0.062, +0.070] | No |
| Older-35plus (Active) | -0.005 | [-0.073, +0.063] | No |
| Young-18-34 (Active) | +0.009 | [-0.073, +0.088] | No |

**Observation:** No aggregate disparities are statistically significant at the 95% level. The Non-Native CI ranges from -0.146 to +0.071 — it crosses zero, meaning we can't conclusively say the aggregate disparity is real. This is a **sample size limitation**: 11 Non-Native annotators is simply not enough to achieve statistical significance at the aggregate level.

However — and this is crucial — this does NOT invalidate the Q6 finding from Experiment 8. The Fisher's exact test on Q6 specifically *is* significant (p=0.006) because the effect size there is enormous (9.1% vs 57.1%). The bootstrap CIs tell us the *overall average* disparity isn't conclusive, but the *question-conditional* disparity is.

---

## Summary of Findings

### Active Sensitive Attributes (Inherent):

**LinguisticGroup (Native vs Non-Native)** — The primary fairness concern.
- Aggregate FairEM: FAIR (TPR disparity -0.088, just below 0.1 threshold)
- Conditional on Hard Questions: **UNFAIR** (accuracy disparity -0.126)
- Q6 specifically: **Severely UNFAIR** (disparity -0.392, Fisher p=0.006)
- Pattern reversal on easy questions: Non-Native actually outperforms
- Calibration: **CALIB-UNFAIR** (+0.126 gap — overconfident)
- Behavioral: Longer decision times, fewer clicks, lower explanation rate

**AgeGroup (Young vs Older)** — No fairness concerns.
- All metrics well within threshold
- Older-35plus has higher calibration gap (+0.110, CALIB-UNFAIR) but accuracy differences are negligible

### Passive Sensitive Attributes (Acquired):

**ExpertiseGroup (STEM vs Non-STEM)** — No fairness concerns.
- Essentially identical performance across all metrics
- STEM slightly more overconfident (CALIB-UNFAIR at +0.110)

**ExperienceGroup (High-Edu vs Lower-Edu)** — No fairness concerns.
- Minimal accuracy differences
- High-Edu has worst calibration (+0.148, CALIB-UNFAIR) — highest confidence, not highest accuracy

### The Core Story:

The bias in this annotation study is **not a broad demographic effect**. It is a **specific interaction between one active sensitive attribute (English proficiency) and task difficulty**. Non-Native English speakers perform just as well — or even better — on straightforward questions. But when questions are linguistically complex or ambiguous (particularly Q6), their accuracy drops dramatically.

This has direct implications for TUS fairness:
1. Aggregate fairness metrics alone are insufficient — we need difficulty-conditional analysis
2. Q6 may have a **differential item functioning** problem — the question itself may be biased
3. Any fair aggregation scheme must account for question difficulty when weighting annotator responses
4. Confidence-based weighting would amplify existing biases (4 groups are overconfident)

### Cross-Dataset Validation:

The Non-Native lower TPR pattern **replicates** from our earlier TUNE Benchmark analysis (where Non-Native TPR=0.579 vs Native=0.663). However, the TUNE finding that Non-STEM was UNFAIR on TPR (d=+0.138) does **not** replicate in this dataset (d=-0.004). This suggests the linguistic bias signal is robust while the expertise signal was dataset-specific.

---

*End of Speaker Script*
