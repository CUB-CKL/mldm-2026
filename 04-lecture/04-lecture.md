#  Machine Learning and Data Mining

~~~
\subtitle{Anomaly Detection and Robust Methods}
\author{Maxim Borisyak}

\institute{Constructor University Bremen}

\usepackage{algorithm}
\usepackage{booktabs}
\usepackage{algpseudocode}
\usepackage{setspace}
\usepackage{framed}

\DeclareMathOperator*{\E}{\mathbb{E}}
\DeclareMathOperator*{\var}{\mathbb{D}}
\newcommand\D[1]{\var\left[ #1 \right]}

\DeclareMathOperator*{\argmin}{\mathrm{arg\,min}}
\DeclareMathOperator*{\argmax}{\mathrm{arg\,max}}
~~~

## Anomaly Detection

### What is an anomaly?

**Anomaly** (outlier): an observation that deviates markedly from the majority of the data.

**Applications**: fraud detection, network intrusion, fault detection, medical diagnosis.

**Types**:
- **point**: a single observation is anomalous (e.g.\ a \$10\,000 transaction among \$20--\$200 ones);
- **contextual**: anomalous in context but normal globally (e.g.\ $30^\circ$C in January);
- **collective**: a group of individually normal observations is anomalous together.

### Problem formulations

**Unsupervised**: only unlabeled data; score each point by deviation from the bulk.

**Semi-supervised (one-class)**: train on clean normal data; flag deviations at test time.

**Supervised**: labeled normal and anomalous examples; standard binary classification, but highly imbalanced.

### Statistical approach: $z$-score

~~~equation*
z_i = \frac{x_i - \mu}{\sigma}
~~~

Flag $x_i$ as outlier if $|z_i| > \tau$ (commonly $\tau = 3$).

**Weakness**: $\mu$ and $\sigma$ are themselves distorted by outliers --- extreme values inflate $\sigma$, masking the anomalies we seek.

### Statistical approach: IQR rule

Let $\mathrm{IQR} = Q_3 - Q_1$. Flag $x_i$ if:
~~~equation*
x_i < Q_1 - 1.5\,\mathrm{IQR} \quad \text{or} \quad x_i > Q_3 + 1.5\,\mathrm{IQR}
~~~

- quartiles are unaffected by extreme values;
- threshold $1.5$ corresponds to $\approx \pm 2.7\sigma$ under Gaussianity.

### Distance-based outlier detection

**$k$-NN score**: $\mathrm{score}(x) = d_k(x)$, the distance to the $k$-th nearest neighbor.

- no distributional assumptions;
- $O(n^2)$ pairwise distances --- slow for large $n$.

**Local Outlier Factor (LOF)**: compares local density of $x$ to that of its neighbors.

~~~equation*
\mathrm{LOF}_k(x) = \frac{1}{k}\sum_{o \in \mathcal{N}_k(x)} \frac{\mathrm{lrd}_k(o)}{\mathrm{lrd}_k(x)}
~~~

$\mathrm{LOF} \approx 1$: normal; $\mathrm{LOF} \gg 1$: anomalous.

## One-Class SVM

### One-class SVM

**Setting**: only normal training data $\{x_1,\dots,x_n\}$.

**Goal**: find a hypersphere in feature space enclosing most of the data; points outside are anomalies.

~~~equation*
\min_{w,\, \xi,\, \rho} \;\frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n}\xi_i
\quad \text{s.t.} \quad
\langle w,\, \phi(x_i)\rangle \geq \rho - \xi_i,\;\; \xi_i \geq 0
~~~

- $\nu \in (0,1]$: upper bound on the outlier fraction and lower bound on support vectors;
- $\rho$: offset from the origin (margin).

### One-class SVM: decision function

~~~equation*
f(x) = \mathrm{sign}\!\left(\sum_{i} \alpha_i\, K(x_i, x) - \rho\right)
~~~

Common choice: RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$.

**Pros**: works in high dimensions; $\nu$ gives direct control over outlier fraction.

**Cons**: $O(n^2)$--$O(n^3)$ training; sensitive to kernel choice; training data must be clean.

## Isolation Forest

### Isolation Forest: key idea

**Principle** (Liu, Ting, Zhou 2008): anomalies are few and different --- they are easier to isolate.

Build random binary trees by repeatedly splitting on a random feature at a random threshold. A point is isolated when it reaches a leaf alone.

- normal point: buried in a dense region, needs many splits;
- anomalous point: in a sparse region, isolated in few splits.

### Isolation Forest: score

Build $t$ trees on subsamples of size $\psi$. **Anomaly score** (normalized to $[0,1]$):

~~~equation*
s(x,\psi) = 2^{-\,\bar{h}(x) / c(\psi)}, \qquad c(\psi) = 2H(\psi - 1) - \frac{2(\psi-1)}{\psi}
~~~

where $\bar{h}(x)$ is the mean path length over the forest and $c(\psi)$ is the expected BST depth.

$s \approx 1$: anomaly; $s \approx 0.5$: normal.

**Complexity**: $O(t\,\psi\log\psi)$ --- linear in $n$, no distance computation.

### Comparing anomaly detectors

~~~center
{\footnotesize\setlength\tabcolsep{4pt}
\begin{tabular}{lllll}
\toprule
\textbf{Method} & \textbf{Complexity} & \textbf{High-dim} & \textbf{Density} & \textbf{Interpretable} \\
\midrule
$z$-score / IQR  & $O(n)$             & No  & No  & Yes \\
$k$-NN / LOF     & $O(n^2)$           & No  & Yes & Partial \\
One-class SVM    & $O(n^2)$--$O(n^3)$ & Yes & Yes & No \\
Isolation Forest & $O(n)$             & Yes & No  & No \\
\bottomrule
\end{tabular}}
~~~

## Robust Regression

### Why OLS fails

**OLS** minimizes $\sum_i r_i^2$ where $r_i = y_i - x_i^\top\beta$.

The squared loss penalizes large residuals quadratically --- a single outlier can dominate the fit.

**Breakdown point**: the fraction of outliers an estimator can tolerate.

- OLS: $1/n \to 0$ (a single outlier suffices to corrupt the estimate).

### Robust loss functions

Replace $r^2$ with $\rho(r)$ that grows slowly for large $|r|$:

~~~center
\begin{tabular}{lll}
\toprule
\textbf{Loss} & $\rho(r)$ & $\psi(r) = \rho'(r)$ \\
\midrule
OLS       & $r^2$                           & $2r$ (unbounded) \\
Huber     & $r^2/2$ or $\delta|r|-\delta^2/2$ & bounded \\
Cauchy    & $\log(1 + r^2/\sigma^2)$        & bounded \\
Tukey     & piecewise cubic                 & redescending (zero for large $|r|$) \\
\bottomrule
\end{tabular}
~~~

### Huber loss and IRLS

~~~equation*
\rho_\delta(r) =
\begin{cases}
r^2/2 & |r| \leq \delta \\
\delta\,|r| - \delta^2/2 & |r| > \delta
\end{cases}
~~~

Quadratic near zero, linear in the tails. $\delta$ is typically set via MAD of residuals.

**IRLS**: minimizing $\sum_i\rho(r_i)$ is equivalent to weighted least squares with adaptive weights $w_i = \psi(r_i)/r_i$:

~~~equation*
\beta^{(t+1)} = \left(X^\top W^{(t)} X\right)^{-1} X^\top W^{(t)} y
~~~

### RANSAC

**Random Sample Consensus**: model-agnostic robust fitting.

~~~framed
\begin{spacing}{1.4}
\begin{algorithmic}[1]
  \Repeat
    \State sample $s$ points; fit model $\hat{\beta}$
    \State find inliers $\mathcal{I} = \{i : |r_i(\hat{\beta})| < \tau\}$
    \State keep $\hat{\beta}$ if $|\mathcal{I}|$ is largest so far
  \Until{$T$ iterations}
  \State refit on all inliers $\mathcal{I}^*$
\end{algorithmic}
\end{spacing}
~~~

Required iterations to succeed with probability $1-\eta$:
$T \geq \log\eta \,/\, \log[1-(1-\epsilon)^s]$, where $\epsilon$ = outlier fraction.

### Theil--Sen estimator

Slope estimated as the **median pairwise slope**:

~~~equation*
\hat{\beta}_1 = \mathrm{median}_{i < j}\; \frac{y_j - y_i}{x_j - x_i}, \qquad
\hat{\beta}_0 = \mathrm{median}_i(y_i - \hat{\beta}_1 x_i)
~~~

- breakdown point $\approx 29\%$;
- $O(n^2)$ naively, $O(n\log n)$ with efficient median algorithms.

## Handling Missing Values

### Types of missingness

**MCAR** (Missing Completely At Random): $P(\text{miss}) $ independent of all values.
- complete-case analysis is unbiased.

**MAR** (Missing At Random): $P(\text{miss} \mid x_\text{obs}, x_\text{miss}) = P(\text{miss} \mid x_\text{obs})$.
- bias unless conditioning on observed variables.

**MNAR** (Missing Not At Random): $P(\text{miss})$ depends on the missing value itself.
- requires modeling the missingness mechanism explicitly.

### Why missingness type matters

~~~center
\begin{tabular}{lp{3.2cm}p{3.2cm}}
\toprule
\textbf{Mechanism} & \textbf{Complete-case} & \textbf{Simple imputation} \\
\midrule
MCAR & Unbiased           & Unbiased \\
MAR  & Biased             & Unbiased if conditioned correctly \\
MNAR & Biased             & Biased \\
\bottomrule
\end{tabular}
~~~

\vspace{3mm}
**Complete-case analysis**: drop rows with any missing value --- simple but wasteful and biased unless MCAR.

### Imputation strategies

**Univariate** (per-column): mean / median / mode / constant.
- distorts variance; ignores correlations.

**$k$-NN imputation**: fill from weighted mean of $k$ nearest complete-case neighbors.

**Regression imputation**: regress $x_j$ on all other features; predict missing entries.

**MICE** (Multiple Imputation by Chained Equations): iterate regression imputation across features.

### Matrix completion

**Low-rank assumption**: $X \approx Z$ with $\mathrm{rank}(Z) \ll d$.

~~~equation*
\min_{Z}\;\sum_{(i,j)\in\Omega} \left(X_{ij} - Z_{ij}\right)^2 + \lambda\|Z\|_*
~~~

Nuclear norm $\|Z\|_* = \sum_j \sigma_j(Z)$ is the convex surrogate for rank. Same machinery as collaborative filtering.

**Native support**: XGBoost, LightGBM, CatBoost send missing-value samples to the child that minimizes loss --- no imputation needed.

### Summary: anomaly detection

- $z$-score / IQR: simple, univariate;
- LOF: density-aware, handles varying density;
- one-class SVM: kernel boundary, high-dimensional;
- Isolation Forest: fast ($O(n)$), scalable.

### Summary: robust regression

- Huber + IRLS: smooth, adaptive down-weighting;
- RANSAC: model-agnostic, handles high outlier fractions;
- Theil--Sen: 29\% breakdown point, simple regression.

### Summary: missing values

- identify mechanism: MCAR / MAR / MNAR;
- MCAR: simple imputation; MAR: MICE or matrix completion;
- MNAR: model the missingness itself;
- tree models handle missing values natively.
