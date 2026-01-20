# Free Entailment Algorithm

## Overview
This algorithm determines semantic entailment between text pairs by combining multiple similarity measures and machine learning classifiers, iteratively improving through LLM feedback.

---

## Setup

### 1. Initial Similarity Generation
- **Generate Cosine Similarity:** Use a bi-encoder SBERT model (currently `BAAI/bge-en-icl`, dependent on resources) to score initial pairings.
- **Select for Labeling:**
  - Pick a fraction of pairs with **high cosine similarity** to send to LLM for labeling.
  - Include a fraction with **low cosine similarity** to ensure diversity for model training.
  - *Note:* In the initial training phase, this step is skipped.

### 2. LLM Labeling
The SBERT model (used later as a cross-encoder) performs **BEST** when the LLM evaluates:
- **A → B?** (Does A entail B?) [*Included in initial training*]
- **B → A?** (Does B entail A?) [*Included in initial training*]
- **Contradiction?** (Does A contradict B and vice versa?)
- **Neutral/Related?** (Are they on the same topic e.g., "The King", but without a logical link?)
- **Paraphrase?** (If A paraphrases B)

**Data Division:**
The data is split into two sets (an initial random sample is used to mimic this):
- **[labeled]**: Pairs with LLM verdicts.
- **[candidates]**: Unlabeled pairs awaiting classification.

**Processing Candidates:**
For each pair $(A, B)$ in [candidates]:
1. Find the equivalence class of $A$ in [labeled], denoted as $[A]$.
2. Find the equivalence class of $B$ in [labeled], denoted as $[B]$.
3. Obtain $\alpha$ weights for similarity scoring.

### 3. Fine-Tune Models on Labeled Data
- **Bi-Encoder:** Fine-tune on all results pairs.
- **Cross-Encoder:** Fine-tune on all results pairs.
  - *Crucial:* The cross-encoder requires `add_cross_encoder_score` to fit the model correctly, as it takes more arguments than the LLM queries.
- **Computation Note:** Training is computationally expensive. Pre-trained models are available in the repository (trained on UChicago RCC). They were trained on the 450,000 LLM results. Please check the notebook comments for instructions on skipping unnecessary computation. Also check documentation found within those libraries to analyze training efficiency.

### 4. Generate Fine-Tuned Similarity Scores
Produce fine-tuned cosine similarity scores for:
- Every pair $(A, B)$ in candidates.
- Every pair $(A, C)$ where $C \in [B]$ (equivalence class of B).
- Every pair $(D, B)$ where $D \in [A]$ (equivalence class of A).

---

## Features

The algorithm uses **4 key features** to predict entailment:

### 1. Cosine Similarity (Bi-Encoder) Neighbor Score
- **Source:** Fine-tuned SBERT bi-encoder.
- **Function:** Weights cosine similarity of $A$ to $[B]$ and $B$ to $[A]$.
- **Formula:**
![equation](https://latex.codecogs.com/svg.image?\color{white}{S_{AB}=\sigma_{AB}\left[\frac{\alpha}{|\mathcal{I}(A)|}\sum_{k\in\mathcal{I}(A)}\sigma_{AC}+\frac{1-\alpha}{|\mathcal{I}(B)|}\sum_{k\in\mathcal{I}(B)}\sigma_{BC}\right]})
  Where:
  - $\sigma_{ij}$ is the cosine similarity of $i$ and $j$.
  - $\mathcal{I}(j) = [j]$ (the equivalence class of $j$, minus $j$ itself).
  - If both $[j]$ and $[i]$ are empty, simply computes $\sigma_{ij}$ as the score.

### 2. NLI Score (Cross-Encoder)
- **Model:** Cross-Encoder (`nli-deberta-v3-base`).
- **Function:** Processes the pair $(A, B)$ jointly defined by the NLI similarity of $A$ to $[B]$ and $B$ to $[A]$.
- **Formula:** Uses the same weighting formula as Feature 1.

### 3. Transitivity Score
- **Method:** Builds a **directed graph** from [labeled] pairs.
- **Check:** Does a path exist $A \leftrightarrow B$ via:
  - $A \to X \to B$ (Forward path)
  - $B \to Y \to A$ (Backward path)
- **Scoring:**
  - **1 hop:** Score = 1.0
  - **2 hops:** Score = $1.0 \times \text{decay}$

### 4. Additional Features
- *TBD*

---

## Predicting Entailment

### 1. Train Multiple Classifiers
The 4 features are fed into the following candidate classifiers:
- **Logistic Regression**
- **Spline Regression**
- **Kernel SVM**
  - Config: `SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)`
- **Decision Tree**
  - Config: `DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=kwargs.get("max_depth", 5))`
- **Gradient Boosting**
  - *Note:* First optimizes parameters with Optuna.
  - Config: `HistGradientBoostingClassifier(random_state=42, class_weight='balanced', learning_rate=0.05, max_iter=200, l2_regularization=1.0)`

### 2. Model Selection
- Selects the model with the **largest ROC-AUC** score.
- **ROC-AUC** measures how well the model distinguishes between "YES" (Entailment) and "NO" (Non-entailment) as you change the threshold ($\tau$).

### 3. Output
- Produces a **probability of entailment** for each candidate pair $(A, B)$.

---

## Sending Back to LLM (Active Learning Loop)

## 1. Find Optimal Cutoff Threshold
Identify the probability threshold ($\tau$) that maximizes one of the following against ground truth labels:
- **Accuracy**
- **F1 Score**

### 2. User-Defined Threshold
Users can select a cutoff based on:
- Precision/Recall tradeoffs.
- Cost of false positives vs. false negatives.
- Domain-specific requirements.

### 2.5. Best Tresholds:
Some useful thresholds are calculated:
- Minimizing FP (auto-accepted but not actually entailed): Calculates F_1/2 using: ![equation](https://latex.codecogs.com/svg.image?\color{white}{F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}}), where precision and recall are calculated using sklearn's precision_recall_curve, and maximizes F_1/2 out of all thresholds.
- Minimizing FN (Sending to LLM when we shouldn't have): Does the same with F_2 as well as calculating Bayes' Optimal Threshold by inputtig the cost of a FN as 5x worse than the cost of an FP (arbitrary, can play with it).

### 3. Active Learning
1. Pairs **above** the threshold are sent to LLM for labeling.
2. Pairs **below** the threshold are automatically rejected as entailments.
3. The cost for doing this is computed (rough average).
4. A dataset is outputed for the pairs to be sent back.
5. **Repeat** the process to iteratively improve model performance.

---

## Technical Requirements

### Installing Imports
Installing the necessary packages can be complex. I uploaded a directory here which has all of the libraries I needed to use in this process. However, I would not be surprised if you need to install a few more things to run this, so just add a cell and %pip install it (and restart the kernel). You will need to figure out the path to this directory on your own and change a few things here and there.

### Training Specs (Reference)
The algorithm was tested and trained on the following hardware configuration:

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Quadro RTX 6000                On  | 00000000:06:00.0 Off |                  Off |
| N/A   30C    P8              13W / 250W |      0MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```
**Warning:** Be cautious if running on hardware with less video memory.
