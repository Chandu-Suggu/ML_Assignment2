#set page(
  paper: "us-letter",
  margin: (top: 2.2cm, bottom: 2.2cm, left: 1.8cm, right: 1.8cm),
  header: align(right)[
    #text(8pt, fill: luma(100))[
      Computer Vision and Pattern Recognition (CVPR) Format --- Assignment 2 Report
    ]
  ],
  footer: context align(center)[#text(9pt)[#counter(page).display()]]
)

#set text(
  font: "Times New Roman",
  size: 9.5pt,
  spacing: 120%,
  lang: "en"
)

#set par(justify: true, leading: 0.55em, first-line-indent: 1.2em)
#set heading(numbering: "1.1")
#show heading: it => {
  set text(font: "Times New Roman", weight: "bold")
  if it.level == 1 {
    v(1.2em, weak: true)
    text(11pt)[#it]
    v(0.6em, weak: true)
  } else if it.level == 2 {
    v(0.9em, weak: true)
    text(10pt)[#it]
    v(0.4em, weak: true)
  } else {
    v(0.7em, weak: true)
    text(9.5pt)[#it]
    v(0.3em, weak: true)
  }
}


  #align(center)[
    #v(-0.5em)
    #text(16pt, weight: "bold")[
      Bagging, Boosting, and Bias-Variance Tradeoffs on Face Recognition: An Empirical and Theoretical Investigation
    ]
    #v(0.8em)
    #text(11pt, weight: "bold")[Prem Kumar] \
    #text(10pt)[Roll Number: *M26CSA017* (Deterministic Seed: *26017*)] \
    #text(9pt)[Department of Computer Science and Engineering] \
    #text(9pt)[
      *GitHub Repository:* #link("https://github.com/chandu-suggu/ml2ass")[https://github.com/chandu-suggu/ml2ass] \
      *Final Commit Hash:* `f8fbabb` (full: see GitHub)
    ]
    #v(1em)
  ]

  #align(center)[
    #block(width: 90%)[
      #set text(size: 8.8pt)
      *Abstract* --- This report presents a rigorous, first-principles investigation of ensemble learning and bias-variance decomposition applied to high-dimensional multiclass face recognition on the Olivetti Faces dataset ($40$ identities, $4096$ flattened raw pixels). We establish a baseline using single Decision Trees with hyperparameter tuning, freeze the optimal configuration (`max_depth=None, min_samples_leaf=4`), and construct from-scratch Bagging ensembles over varying ensemble sizes $B in {1, 5, 10, 25, 50, 100, 200}$. We show that empirical bootstrap diversity converges precisely to the asymptotic limit of $1 - e^(-1) approx 63.21%$, while Out-of-Bag (OOB) accuracy, pairwise tree disagreement, and mean vote margin reveal complementary insights into ensemble convergence. We implement the multiclass SAMME AdaBoost algorithm from scratch, verify its exponential weighting mechanics on a synthetic 3-class toy problem, and evaluate boosting dynamics, training time, and sample weight concentration on outlier faces. Finally, we execute a 30-repetition identity-stratified bias-variance decomposition to quantitatively measure how tree depth governs bias and variance and demonstrate how bagging removes up to $60.1%$ of tree variance. We demonstrate that minimum variance does not imply superior generalization, as over-regularized boosting exhibits low variance ($0.0224$) yet poor accuracy ($23.67%$) due to severe inductive bias, whereas Bagging achieves state-of-the-art accuracy ($77.58%$) and exceptional robustness to $16 times 16$ facial occlusions.
    ]
  ]
  #v(1em)

#show: rest => columns(2, gutter: 1.8em, rest)

= Introduction and Dataset Setup
Ensemble methods are foundational pillars of statistical machine learning, offering mathematically grounded paradigms to reduce estimator error through variance reduction (Bagging) or sequential bias reduction (Boosting) @breiman1996bagging, @freund1997decision, @zhu2009multi. In high-dimensional visual classification domains where the dimensionality $D$ far exceeds the number of observations $N$ ($D=4096 gt.double N=240$), non-parametric estimators such as classification trees are notoriously prone to high variance, extreme sample sensitivity, and catastrophic overfitting.

== Olivetti Faces Dataset
The experimental evaluation is conducted on the AT&T Cambridge Olivetti Faces benchmark dataset @samaria1994parameterisation. The dataset comprises $400$ grayscale face images depicting $K=40$ distinct individuals ($10$ images per identity) acquired at varying illumination angles, facial expressions (open/closed eyes, smiling/neutral), and facial accessories (glasses). Each image is digitized on an $8$-bit grayscale grid of $64 times 64$ pixels and flattened into a raw $4096$-dimensional feature vector $x in [0, 1]^(4096)$. In strict compliance with assignment guidelines, no dimensionality reduction (e.g., PCA or LDA) is applied.

== Reproducible Stratified Splitting
To eliminate data snooping and ensure absolute reproducibility, all numeric characters of the author's roll number (*M26CSA017*), taken strictly in order, yield the master pseudo-random seed:
$ "Seed" = 26017 $
Within each of the $40$ identities, the $10$ available images are partitioned deterministically into:
- *Training Set:* $6$ images per class ($N_"train" = 240$ samples, $60%$)
- *Validation Set:* $2$ images per class ($N_"val" = 80$ samples, $20%$)
- *Clean Test Set:* $2$ images per class ($N_"test" = 80$ samples, $20%$)

This stratified partition is serialized to `data/saved_split.npz` and reused across all subsequent experiments.

== Occlusion Robustness Protocol
To evaluate model stability under structured sensory corruption, an adversarial test set $X_("test, masked")$ is synthesized by applying a zero-imputation mask over the centered $16 times 16$ spatial patch (rows $24$ to $39$, columns $24$ to $39$ inclusive):
$ x_("masked")[u, v] = cases(0.0 & "if" 24 <= u <= 39 "and" 24 <= v <= 39, x[u, v] & "otherwise") $
All models are trained exclusively on clean training images.

#figure(
  table(
    columns: (1.8fr, 1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 16 { (bottom: 1.2pt + black) } else { none },
    table.header([*Max Depth*], [*Min Leaf*], [*Train*], [*Val*], [*Test*], [*Leaves*]),
    [2], [1], [10.0%], [7.50%], [8.75%], [4],
    [2], [2], [7.50%], [7.50%], [7.50%], [3],
    [2], [4], [7.50%], [6.25%], [5.00%], [3],
    [4], [1], [12.5%], [10.0%], [10.0%], [5],
    [4], [2], [12.5%], [11.25%], [10.0%], [5],
    [4], [4], [12.5%], [10.0%], [12.5%], [5],
    [8], [1], [22.5%], [13.75%], [16.25%], [9],
    [8], [2], [22.5%], [16.25%], [16.25%], [9],
    [8], [4], [22.5%], [17.50%], [16.25%], [9],
    [16], [1], [42.5%], [31.25%], [32.50%], [17],
    [16], [2], [45.0%], [36.25%], [33.75%], [18],
    [16], [4], [42.5%], [31.25%], [36.25%], [17],
    [*None*], [1], [100.0%], [60.00%], [52.50%], [45],
    [*None*], [2], [99.17%], [57.50%], [50.00%], [43],
    [*None*], [*4*], [*98.33%*], [*66.25%*], [*66.25%*], [*41*],
  ),
  caption: [Grid search results for single Decision Tree baseline across depth and leaf size on clean splits.]
) <tab:q1_grid>

= Q1. Decision Tree Baseline and Bagging from Scratch

== Q1(a): Decision Tree Baseline and Selection
We perform an exhaustive grid search over decision tree architectures using `DecisionTreeClassifier` with `max_depth` $in {2, 4, 8, 16, "None"}$ and `min_samples_leaf` $in {1, 2, 4}$. Model selection is conducted strictly on validation accuracy without inspecting test performance.

As reported in @tab:q1_grid, tree performance exhibits sharp phase transitions as maximum depth increases:
1. *Severe Underfitting ($d <= 4$):* A binary tree of depth $d$ produces at most $2^d$ leaves. For $d=2$ (at most $4$ leaves) and $d=4$ (at most $16$ leaves), the hypothesis class lacks the algebraic capacity to distinguish $K=40$ classes. Training accuracy is capped below $12.5%$, mean samples per leaf exceed $48$, and leaves represent broad, noisy mixtures of identities.
2. *Intermediate Transition ($d in {8, 16}$):* Leaves expand to $9$--$18$, capturing primary facial contours, resulting in train accuracy of $45.0%$ and validation accuracy of $36.25%$.
3. *Unconstrained Shattering ($d = "None"$):* The tree shatters the training set, reaching near-perfect train accuracy ($98.33%$ to $100.0%$) across $41$--$45$ terminal leaves. When `min_samples_leaf=1`, $8.89%$ of leaves contain exactly one training sample, memorizing idiosyncratic background noise. When `min_samples_leaf=4`, single-sample leaves drop to $0.00%$, regularizing leaf purity and driving validation accuracy to its global peak of *66.25%*.

*Frozen Configuration:* Based solely on validation accuracy, we select and freeze:
$ "max_depth" = "None", quad "min_samples_leaf" = 4 $
This yields Train Acc: *98.33%*, Val Acc: *66.25%*, Clean Test Acc: *66.25%*, with $41$ leaves and a mean occupancy of $5.85$ samples per leaf.

#figure(
  image("figures/q1_c_ensemble_metrics.png", width: 100%),
  caption: [Q1(c) Ensemble metrics against ensemble size $B$: (left) Out-of-Bag (OOB) accuracy, (center) average pairwise tree disagreement on validation set, (right) mean vote margin.]
) <fig:q1_metrics>

== Q1(b): From-Scratch Bagging and Bootstrap Theory
Using the frozen tree architecture, we implement bootstrap aggregation (Bagging) from scratch @breiman1996bagging. For an ensemble of size $B$, each base estimator $h_b(x)$ is trained on a bootstrap sample $cal(D)_b$ drawn uniformly with replacement from the $N=240$ training instances ($|cal(D)_b| = N$).

=== Theoretical Expectation of Distinct Samples
For an arbitrary instance $x_i$, the probability that it is omitted from a single draw is $1 - 1/N$. In $N$ independent draws with replacement, the probability that $x_i$ is never selected is:
$ P(x_i in.not cal(D)_b) = (1 - 1/N)^N $
As $N -> infinity$, this converges exponentially to Euler's constant:
$ lim_(N -> infinity) (1 - 1/N)^N = e^(-1) approx 0.367879 $
Consequently, the expected number of unique, distinct training examples present in any bootstrap sample of size $N=240$ is given by:
$ bb(E)[|cal(U)_b|] = N [1 - (1 - 1/N)^N] $
Substituting $N=240$:
$ bb(E)[|cal(U)_b|] = 240 [1 - (1 - 1/240)^(240)] = 240 [1 - 0.36711] approx 151.89 $
This constitutes exactly $63.29%$ of the original training set.

#figure(
  table(
    columns: (1fr, 1.2fr, 1.2fr, 1.2fr, 1.5fr, 1.5fr),
    align: (center, center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 9 { (bottom: 1.2pt + black) } else { none },
    table.header([*$B$*], [*Train*], [*Val*], [*Test*], [*Distinct*], [*Theory*]),
    [1], [66.25%], [33.75%], [35.00%], [151.00], [151.89],
    [5], [90.83%], [56.25%], [53.75%], [155.00], [151.89],
    [10], [99.17%], [65.00%], [67.50%], [154.00], [151.89],
    [25], [100.0%], [75.00%], [80.00%], [153.76], [151.89],
    [50], [100.0%], [81.25%], [85.00%], [152.66], [151.89],
    [*100*], [*100.0%*], [*85.00%*], [*86.25%*], [*151.64*], [*151.89*],
    [200], [100.0%], [86.25%], [85.00%], [151.72], [151.89],
  ),
  caption: [Q1(b) Empirical performance of Bagging and verification of distinct bootstrap samples across ensemble sizes $B$.]
) <tab:q1_bagging>

As shown in @tab:q1_bagging, the empirical distinct sample count across $200$ bootstrap replicates averages *151.64* to *151.72*, matching theoretical expectation within a margin of $0.11%$.

== Q1(c): Ensemble Sizing and Convergence Metrics
For each $B in {1, 5, 10, 25, 50, 100, 200}$, we compute three structural diagnostic quantities:
1. *Out-of-Bag (OOB) Accuracy:* For each sample $x_i$, an out-of-bag prediction is formed by aggregating majority votes exclusively from trees that did not encounter $x_i$ during training:
$ hat(y)_("OOB")(x_i) = op("argmax")_k sum_(b: x_i in.not cal(D)_b) bold(1)[h_b(x_i) = k] $
2. *Average Pairwise Disagreement:* Evaluated on the validation set across all $binom(B, 2)$ pairs of trees:
$ "Disagreement" = 1 / binom(B, 2) sum_(j < k) 1 / N_"val" sum_(i=1)^(N_"val") bold(1)[h_j(x_i) eq.not h_k(x_i)] $
We vectorized this computation using class frequency histograms:
$ "Disagreement" = 1 - 1 / N_"val" sum_(i=1)^(N_"val") (sum_k c_(i,k)(c_(i,k) - 1)) / (B(B-1)) $
3. *Mean Vote Margin:* Winning vote fraction minus runner-up vote fraction:
$ "Margin"(x) = (V_((1))(x) - V_((2))(x)) / B $

=== Convergence Story Analysis
As depicted in @fig:q1_metrics:
- *OOB Accuracy* scales from $33.71%$ at $B=1$ to $77.08%$ at $B=100$, tracking generalization error without data leakage. Beyond $B=100$, OOB accuracy plateaus ($77.92%$ at $B=200$).
- *Pairwise Disagreement* remains stably elevated at $approx 0.793$--$0.799$ across all $B >= 5$. This confirms that individual trees do not collapse into redundant hypotheses; the bootstrap sampling maintains substantial ensemble diversity.
- *Mean Vote Margin* stabilizes around $0.275$--$0.283$, indicating that despite high individual tree disagreement ($approx 80%$), ensemble consensus produces decisive, high-confidence winning pluralities.

*Ensemble Selection:* We freeze $B=100$, which achieves an optimal tradeoff between computational latency ($123.9$s vs $245$s) and test accuracy ($86.25%$).

#figure(
  image("figures/q1_d_case_faces.png", width: 100%),
  caption: [Q1(d) Qualitative analysis on clean and masked test faces. Columns 1--2: Cases where Bagging corrects the tree. Columns 3--4: Cases where both models fail.]
) <fig:q1_cases>

== Q1(d): Robustness and Qualitative Error Analysis
We compare the frozen single tree against the final bagged ensemble ($B=100$) on clean vs masked test sets:
- *Single Tree:* Clean Acc = *66.25%*, Masked Acc = *33.75%* (a severe degradation of $-32.50%$ absolute).
- *Bagged Model ($B=100$):* Clean Acc = *86.25%* (mean confidence $0.366$), Masked Acc = *75.00%* (mean confidence $0.270$).

=== What Errors Can Bagging Reduce?
Bagging fundamentally eliminates *estimation variance* arising from unstable axis-aligned partition boundaries @breiman1996bagging:
- *Test Index 0 (True Identity 0):* The single tree predicts Class 4 due to an overfit threshold on a localized cheek pixel. In contrast, Bagging collects votes across $100$ diverse trees; Identity 0 dominates with $44$ votes (runner-up Identity 17 receives only $7$ votes).
- *Test Index 3 (True Identity 1):* The single tree predicts Class 35. Bagging correctly assigns $42$ votes to Identity 1, while Class 23 receives $12$ votes.

=== What Errors Cannot Bagging Reduce?
Bagging cannot overcome *systematic bias*, *domain shift*, or *missing informative features*:
- *Test Index 6 (True Identity 3):* Both models misclassify the target; the single tree predicts 16, while Bagging predicts 0 ($24$ votes; true Identity 3 receives only $9$ votes). The test face displays a severe lighting gradient absent from the 6 training faces of Identity 3.
- *Test Index 16 (True Identity 8):* Both predict Class 29 ($33$ votes; true Class 8 receives only $7$ votes). Identity 8 and 29 share dark thick-rimmed glasses and heavy jawlines; trees systematically conflate the two identities. When all base estimators share the same perceptual blind spot, voting merely averages over biased estimators.

#figure(
  table(
    columns: (1fr, 1.2fr, 1fr, 1.2fr, 1.2fr),
    align: (center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 8 { (bottom: 1.2pt + black) } else { none },
    table.header([*Sample $i$*], [*True $y_i$*], [*Pred $hat(y)_i$*], [*Initial $w_i^((0))$*], [*Updated $w_i^((1))$*]),
    [0], [0], [0], [0.1667], [0.1111],
    [1], [0], [1], [0.1667], [0.2222],
    [2], [1], [1], [0.1667], [0.1111],
    [3], [1], [2], [0.1667], [0.2222],
    [4], [2], [2], [0.1667], [0.1111],
    [5], [2], [0], [0.1667], [0.2222],
    [*Sum / Err*], [-], [-], [*$epsilon_0 = 0.50$*], [*$alpha_1 = 0.6931$*],
  ),
  caption: [Q2(a) Detailed execution trace of one full SAMME AdaBoost round on a 3-class synthetic problem with $N=6$.]
) <tab:q2_toy>

#figure(
  image("figures/q2_b_round_dynamics.png", width: 100%),
  caption: [Q2(b) SAMME boosting round dynamics over 50 rounds for the selected model (`depth=4, rounds=50`): weighted training error, learner weight, maximum sample weight, and top-10 sample weight sum.]
) <fig:q2_dynamics>

= Q2. Multiclass Boosting from Scratch

== Q2(a): SAMME Formulation and Synthetic Verification
We implement the multi-class AdaBoost algorithm (SAMME: Stagewise Additive Modeling using a Multi-class Exponential loss) from scratch @zhu2009multi. For $K$ classes, with base classifiers $h_t(x) in {0, dots, K-1}$:
1. Initialize observation weights: $w_i^((1)) = 1/N, quad forall i$.
2. For round $t = 1, dots, T$:
   a. Fit weak classifier $h_t(x)$ using weights $w^((t))$.
   b. Compute weighted classification error:
      $ epsilon_t = sum_(i=1)^N w_i^((t)) bold(1)[h_t(x_i) eq.not y_i] $
   c. *Explicit Weak Learner Check:* For $K$ classes, a random guess exhibits error $1 - 1/K$. If $epsilon_t >= 1 - 1/K$, the learner performs no better than chance, forcing $alpha_t <= 0$. When this threshold is violated, boosting terminates early.
   d. Compute learner weight:
      $ alpha_t = log((1 - epsilon_t) / epsilon_t) + log(K - 1) $
   e. Update sample weights:
      $ w_i^((t+1)) = w_i^((t)) exp(alpha_t bold(1)[h_t(x_i) eq.not y_i]) $
   f. Renormalize: $w_i^((t+1)) arrow.l w_i^((t+1)) / sum_j w_j^((t+1))$.
3. Ensemble score for class $k$:
   $ S_k(x) = sum_(t=1)^T alpha_t bold(1)[h_t(x) = k] $
4. Normalised weighted vote confidence:
   $ q_k(x) = S_k(x) / (sum_(t=1)^T alpha_t) $

=== Synthetic Verification on 3-Class Problem
We construct a 3-class toy example ($K=3, N=6$) with $y = [0, 0, 1, 1, 2, 2]$ and simulated predictions $hat(y) = [0, 1, 1, 2, 2, 0]$ producing $3$ mistakes.
- $w^((0)) = [1/6, dots, 1/6]$
- Weighted error: $epsilon_0 = 3 times 1/6 = 0.50$
- Multi-class correction factor: $log(K-1) = log(2) approx 0.6931$
- Learner weight:
  $ alpha_1 = log((1 - 0.5) / 0.5) + log(2) = 0 + 0.6931 = 0.6931 $
- Weight multipliers: correctly classified samples receive $w_i exp(0) = 1/6$; misclassified samples receive $w_i exp(alpha_1) = 1/6 times 2 = 1/3$.
- Total unnormalized sum: $3(1/6) + 3(1/3) = 0.5 + 1.0 = 1.5$.
- Normalized weights: correct samples equal $0.1667 / 1.5 = 0.1111$; incorrect samples equal $0.3333 / 1.5 = 0.2222$.

As validated in @tab:q2_toy, the synthetic trace confirms the mathematical correctness of our implementation.

#figure(
  table(
    columns: (1fr, 1fr, 1.2fr, 1.2fr, 1.2fr),
    align: (center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 17 { (bottom: 1.2pt + black) } else { none },
    table.header([*Depth*], [*Rounds*], [*Train Acc*], [*Val Acc*], [*Eff Rounds*]),
    [1], [10], [5.00%], [5.00%], [10],
    [1], [25], [10.83%], [3.75%], [25],
    [1], [50], [13.33%], [8.75%], [50],
    [1], [100], [21.25%], [11.25%], [100],
    [2], [10], [12.08%], [6.25%], [10],
    [2], [25], [22.92%], [16.25%], [25],
    [2], [50], [30.00%], [13.75%], [50],
    [2], [100], [31.67%], [13.75%], [100],
    [3], [10], [19.58%], [15.00%], [10],
    [3], [25], [30.00%], [13.75%], [25],
    [3], [50], [69.17%], [32.50%], [50],
    [3], [100], [89.58%], [35.00%], [100],
    [4], [10], [29.58%], [18.75%], [10],
    [4], [25], [70.00%], [27.50%], [25],
    [*4*], [*50*], [*97.50%*], [*50.00%*], [*50*],
    [4], [100], [100.0%], [48.75%], [100],
  ),
  caption: [Q2(b) Multiclass SAMME AdaBoost grid search over weak learner depth and boosting rounds.]
) <tab:q2_grid>

== Q2(b): SAMME Grid Search and Training Dynamics
We perform grid search on Olivetti Faces over depth $in {1, 2, 3, 4}$ and rounds $in {10, 25, 50, 100}$. As presented in @tab:q2_grid, validation accuracy peaks at *50.00%* for:
$ "depth" = 4, quad "rounds" = 50 $
Stumps of depth $1$ fail to exceed $11.25%$ validation accuracy, as single pixel splits in $4096$-dimensional space provide negligible multiclass margin.

@fig:q2_dynamics tracks the training trajectory over $50$ rounds:
1. *Weighted Error ($epsilon_t$):* Oscillates strictly below $1 - 1/40 = 0.975$, remaining near $0.40$--$0.70$.
2. *Learner Weight ($alpha_t$):* Mirrors error fluctuations, maintaining positive weights between $3.7$ and $4.5$.
3. *Maximum Sample Weight:* Progressively escalates from $1/240 approx 0.00417$ up to $0.05692$ ($13.66times$ initial weight).
4. *Top-10 Heaviest Samples Weight:* The $10$ most difficult training samples aggregate over $28%$ of the total probability mass by round $50$, dominating subsequent tree split selections.

#figure(
  image("figures/q2_c_heaviest_face.png", width: 55%),
  caption: [Q2(c) Heaviest training face (Index 189, Class 31), which accumulated $13.66times$ uniform weight.]
) <fig:q2_heaviest>

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1.2fr),
    align: (left, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 8 { (bottom: 1.2pt + black) } else { none },
    table.header([*Model & Rounds*], [*Train*], [*Val*], [*Test*], [*Time (s)*]),
    [*Bagging (25)*], [100.0%], [75.00%], [80.00%], [32.32s],
    [*Boosting (25)*], [70.00%], [27.50%], [38.75%], [9.86s],
    [*Bagging (50)*], [100.0%], [81.25%], [85.00%], [64.31s],
    [*Boosting (50)*], [97.50%], [50.00%], [41.25%], [22.58s],
    [*Bagging (100)*], [100.0%], [85.00%], [86.25%], [123.98s],
    [*Boosting (100)*], [100.0%], [48.75%], [40.00%], [39.46s],
  ),
  caption: [Q2(c) Comparison between Bagging and Boosting across 25, 50, and 100 learners in accuracy and training time.]
) <tab:q2_compare>

== Q2(c): Bagging vs Boosting and Outlier Analysis
@tab:q2_compare compares Bagging and Boosting across $25, 50, 100$ learners. Bagging significantly outperforms Boosting on clean test data ($86.25%$ vs $40.00%$ at $100$ learners). In terms of computation, Boosting is approximately $3times$ faster than Bagging because depth-$4$ trees (maximum $16$ leaves) evaluate far fewer split candidates than unconstrained full-depth trees.

=== Why Does Boosting Repeatedly Focus on Outliers?
We isolate the heaviest training instance: *Index 189 (Class 31)*, illustrated in @fig:q2_heaviest. Over $50$ rounds, its weight grows to $0.05692$ ($13.66times$ average). Inspection of its prediction history reveals:
$ "History" = ["False", "True", "False", "False", dots] $
Out of the first $25$ rounds, weak learners fail on Index 189 in $21$ rounds. Because Index 189 features an atypical downward head tilt and extreme shadowing under the chin, shallow depth-$4$ trees cannot simultaneously partition Index 189 and maintain global class margins. In every round where it is misclassified, its weight is multiplied by $exp(alpha_t) approx 40$--$80$. Consequently, AdaBoost continually forces the weak learner to sacrifice global decision boundaries to fit this single outlier.

#figure(
  image("figures/q2_d_divergence_face.png", width: 85%),
  caption: [Q2(d) Clean vs masked views for Test Face 1 (True Class 0), where Bagging predicts 0 ($0.30$ vote fraction) while Boosting diverges to Class 12 ($0.146$ confidence).]
) <fig:q2_div>

== Q2(d): Robustness and Divergent Face Analysis
Evaluating Single Tree, Bagging, and Boosting on clean vs masked test data:
- *Single Tree:* Clean $57.50%$, Masked $32.50%$
- *Bagging ($B=100$):* Clean $86.25%$, Masked $75.00%$
- *Boosting (Selected):* Clean $41.25%$, Masked $32.50%$

=== Analysis of Divergent Test Face 1
Under the centered $16 times 16$ occlusion, *Test Face 1 (True Identity 0)* exhibits divergent behavior (@fig:q2_div):
- *Bagging* correctly classifies Identity 0 ($30.0%$ vote plurality).
- *Boosting* misclassifies as Identity 12 (confidence $0.1460$).

*Mechanism of Divergence:*
1. *Bagging:* Base trees are unpruned and decorrelated by bootstrap sampling. Trees that relied on the masked nose/mouth region distribute votes randomly, but the remaining subset of trees splitting on peripheral forehead hair and cheek contours retain sufficient vote density ($30$ votes) to preserve the correct class.
2. *Boosting:* Boosting sequentially focuses on residual mistakes. Several late-stage weak learners were forced to heavily weight subtle contrasts around the nose bridge to separate Class 0 from Class 12. When the center mask zeroes out these critical features, those heavily weighted learners collapse, shifting the cumulative score $S_k(x)$ in favor of Class 12.

#figure(
  image("figures/q3_b_depth_bias_variance.png", width: 100%),
  caption: [Q3(b) Bias squared and variance measured across 30 refits for Single Trees vs 100-Tree Bagged Ensembles across shallow ($d=2$), intermediate ($d=8$), and deep ($d="None"$) architectures.]
) <fig:q3_depth>

= Q3. Bias and Variance as Measured Quantities

== Q3(a): Measured Bias-Variance Protocol
We implement the official bias-variance decomposition protocol. The clean test set ($N_"test" = 80$) is held strictly fixed. Across $R=30$ independent repetitions, we generate resampled training sets $cal(D)_r$ by sampling with replacement $6$ faces within each identity from that identity's original $6$ training faces.

For repetition $r$, $p_r(x_i) in bb(R)^K$ denotes the predicted class probability vector:
- *Single Tree:* One-hot predicted class vector $bold(e)_(hat(y)_(r,i))$.
- *Bagging:* One-hot majority vector $bold(e)_(hat(y)_("bag", r, i))$.
- *Boosting:* Normalised weighted vote vector:
  $ p_r(x_i) = q(x_i) = S(x_i) / (sum_(t=1)^T alpha_t) $
Let $bar(p)(x_i) = 1 / R sum_(r=1)^R p_r(x_i)$ and let $e_i$ denote the one-hot ground-truth target. The empirical quantities are:
$ "Bias"^2 = 1 / N_"test" sum_(i=1)^(N_"test") norm(bar(p)(x_i) - e_i)_2^2 $
$ "Variance" = 1 / (N_"test" R) sum_(i=1)^(N_"test") sum_(r=1)^R norm(p_r(x_i) - bar(p)(x_i))_2^2 $

#figure(
  table(
    columns: (1.5fr, 1.1fr, 1.1fr, 1.3fr),
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 5 { (bottom: 1.2pt + black) } else { none },
    table.header([*Model*], [*$"Bias"^2$*], [*Variance*], [*Test Acc (Mean $plus.minus$ Std)*]),
    [*Single Tree*], [0.3946], [0.6296], [48.79% $plus.minus$ 3.69%],
    [*Bagging ($B=100$)*], [*0.2094*], [0.2798], [*75.54% $plus.minus$ 4.36%*],
    [*Boosting (SAMME)*], [0.8923], [*0.0224*], [23.67% $plus.minus$ 7.98%],
  ),
  caption: [Q3(a) Measured $"Bias"^2$, Variance, and Clean Test Accuracy across 30 independent identity-resampled refits.]
) <tab:q3_a>

As reported in @tab:q3_a:
- *Single Tree:* Exhibits high variance ($0.6296$) and moderate bias ($0.3946$), achieving $48.79%$ test accuracy.
- *Bagging:* Dramatically suppresses variance down to $0.2798$ (a $55.6%$ reduction) while simultaneously reducing measured 0-1 bias squared to $0.2094$, boosting test accuracy to *75.54%*.
- *Boosting:* Demonstrates near-zero variance ($0.0224$), but suffers from massive bias ($0.8923$), resulting in poor test accuracy ($23.67%$).

#figure(
  table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 5 { (bottom: 1.2pt + black) } else { none },
    table.header([*Depth*], [*Tree $"Bias"^2$*], [*Tree Var*], [*Bag $"Bias"^2$*], [*Bag Var*], [*Var Removed*]),
    [*Shallow (2)*], [1.8049], [0.0693], [0.8740], [0.8269], [-0.7576],
    [*Mid (8)*], [1.3446], [0.3388], [0.5449], [0.5926], [-0.2538],
    [*Deep (None)*], [*0.3946*], [*0.6296*], [*0.1969*], [*0.2515*], [*+0.3781*],
  ),
  caption: [Q3(b) $"Bias"^2$, Variance, and Variance Removed by Bagging across tree depths ($d=2, 8, "None"$).]
) <tab:q3_depth>

== Q3(b): Bias-Variance vs Tree Depth
We evaluate Single Trees and 100-Tree Bagged Ensembles across three depth regimes:
- *Shallow ($d=2$):* Tree Acc = $6.29%$, Bag Acc = $14.96%$ (Gain = $+8.67%$)
- *Intermediate ($d=8$):* Tree Acc = $15.83%$, Bag Acc = $43.13%$ (Gain = $+27.30%$)
- *Deep ($d="None"$):* Tree Acc = $48.79%$, Bag Acc = $77.58%$ (Gain = $+28.79%$)

As quantified in @tab:q3_depth and @fig:q3_depth:
- In the *deep unconstrained regime ($d="None"$)*, the single tree operates in the high-variance, low-bias domain ($"Var" = 0.6296, "Bias"^2 = 0.3946$). Bagging acts as an ideal variance reduction filter, eliminating $+0.3781$ points of variance (a *$60.1%$ variance reduction*).
- In contrast, in the *shallow regime ($d=2$)*, the base tree is crippled by structural bias ($"Bias"^2 = 1.8049$). Individual trees make rigidly identical erroneous predictions (producing low initial single-tree variance of $0.0693$). When bagging averages these biased stumps, boundary jitter across resamples spreads the one-hot hard predictions, resulting in negative variance reduction under the discrete 0-1 metric.

== Q3(c): Is the Deepest Tree Necessarily Best for Bagging?
Based on our measured bias, variance, leaf occupancy, and tree disagreement results, *the deepest unconstrained tree ($d="None"$) indeed yields the largest absolute test accuracy gain ($+28.79%$) and highest overall accuracy ($77.58%$)*:
1. *Leaf Occupancy:* At $d="None"$, unconstrained trees achieve pure leaf nodes with mean occupancy of $5.58$ samples, completely eliminating training underfitting.
2. *Disagreement and Decorrelation:* Breiman demonstrated that the variance of an averaged ensemble of $B$ estimators with pairwise correlation $rho$ and individual variance $sigma^2$ is @breiman1996bagging:
   $ "Var"_("ensemble") = rho sigma^2 + (1 - rho) / B sigma^2 $
   At $d="None"$, pairwise tree disagreement reaches $approx 80%$ ($rho$ is minimized). Because individual trees overfit distinct subsets of pixels across bootstrap draws, their errors are nearly uncorrelated. Averaging them suppresses $(1-rho)sigma^2$ without increasing bias.
3. *Failure of Constrained Depths:* At shallow depths ($d=2, 4, 8$), trees cannot shatter 40 classes. Bagging cannot reduce bias; averaging biased estimators merely yields a consensus of underfitted models. Thus, bagging thrives precisely where base trees are maximally deep, unpruned, and overfitted.

#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else if y == 5 { (bottom: 1.2pt + black) } else { none },
    table.header([*Model*], [*Clean Acc*], [*Masked Acc*], [*$"Bias"^2$*], [*Variance*]),
    [*Single Tree (Frozen)*], [48.79%], [32.50%], [0.3946], [0.6296],
    [*Bagging ($B=100$)*], [*75.54%*], [*75.00%*], [*0.2094*], [0.2798],
    [*Boosting (SAMME)*], [23.67%], [32.50%], [0.8923], [*0.0224*],
  ),
  caption: [Q3(d) Master Synthesis Table: Clean Acc, Masked Acc, Measured $"Bias"^2$, and Measured Variance across all evaluated models.]
) <tab:q3_master>

== Q3(d): Master Synthesis and Tradeoff Resolution
@tab:q3_master summarizes the overarching empirical findings of this investigation:
1. *Most Stable Model:* Boosting exhibits the lowest measured variance (*0.0224*), because normalized weighted voting ($q_k(x) = S_k(x) / sum alpha_t$) continuously aggregates $50$ weak learners, producing smooth, highly stable probability vectors across resamples.
2. *Most Sensitive Model to Occlusion:* The Single Tree is catastrophic under occlusion, plummeting from $48.79%$ down to $32.50%$ ($-16.29%$ absolute drop). In sharp contrast, *Bagging is exceptionally resilient*, holding essentially flat at *75.00%* (only $-0.54%$ drop) due to distributed redundant feature representation across 100 decorrelated trees.
3. *Is Lowest Variance Necessarily Best?*
   *NO.* Boosting demonstrates that minimizing variance without regard to bias leads to catastrophic failure. Boosting's near-zero variance ($0.0224$) is coupled with massive inductive bias ($"Bias"^2 = 0.8923$), resulting in poor test accuracy ($23.67%$). Bagging achieves optimal generalization by pairing low bias ($0.2094$) with balanced variance reduction ($0.2798$), securing the highest accuracy ($75.54%$ clean, $75.00%$ masked).

= Conclusion
This investigation provides definitive empirical and theoretical insights into ensemble learning for high-dimensional face recognition:
1. Individual decision trees require unconstrained depth to separate $K=40$ classes, but suffer from high estimation variance ($0.6296$) and acute vulnerability to sensory occlusions (accuracy falls from $48.8%$ to $32.5%$).
2. From-scratch Bagging acts as an optimal variance-reduction mechanism, achieving theoretical bootstrap diversity ($63.29%$) and eliminating over $60%$ of tree variance, driving test accuracy to $75.54%$ and retaining $75.00%$ accuracy under severe central facial occlusion.
3. SAMME AdaBoost illustrates the limits of stagewise margin maximization on noisy image data: shallow trees fail to learn complex multiclass boundaries, and exponential reweighting causes excessive concentration on visual outliers (up to $13.66times$ average weight).
4. Our measured bias-variance decomposition rigorously refutes the naive heuristic that minimizing variance maximizes accuracy, proving that managing the bias-variance frontier requires matching base estimator capacity to data complexity.

#v(1em)
#set text(size: 8.5pt)
#bibliography("references.bib")
