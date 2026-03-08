#  Machine Learning and Data Mining

~~~
\subtitle{Recommendation Systems}
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

\newcommand\dmid{\,\|\,}

\DeclareMathOperator*{\argmin}{\mathrm{arg\,min}}
\DeclareMathOperator*{\argmax}{\mathrm{arg\,max}}
~~~

## Recommendation Systems

### What are recommendation systems?

**Goal**: predict user preferences for items they have not yet interacted with.

**Applications**:
- movies: Netflix, YouTube;
- music: Spotify, Last.fm;
- products: Amazon, Alibaba;
- news, social media, advertisements.

**Key challenge**: the rating matrix is extremely sparse --- most users interact with a tiny fraction of all items.

### Problem formulation

**Setting**:
- $m$ users: $u \in \{1, \dots, m\}$;
- $n$ items: $i \in \{1, \dots, n\}$;
- rating matrix $R \in \mathbb{R}^{m \times n}$, where $R_{ui}$ is the rating of user $u$ for item $i$;
- $\Omega \subseteq \{1,\dots,m\} \times \{1,\dots,n\}$ --- set of observed (user, item) pairs.

**Goal**: estimate $R_{ui}$ for all $(u, i) \notin \Omega$.

### Rating matrix

~~~equation*
R = \begin{pmatrix}
  5 & \cdot & 3 & \cdot & 1 \\
  4 & \cdot & \cdot & 1 & \cdot \\
  \cdot & 2 & \cdot & 4 & \cdot \\
  \cdot & \cdot & 5 & \cdot & 2 \\
  1 & \cdot & 4 & \cdot & 5
\end{pmatrix}
~~~

- rows = users; columns = items;
- $R_{ui} \in \{1,\dots,5\}$ or missing ($\cdot$);
- goal: fill in the missing entries.

### Types of feedback

**Explicit feedback**:
- user actively rates an item (1--5 stars, thumbs up/down);
- sparse but directly expresses preference;
- examples: movie ratings, book reviews.

**Implicit feedback**:
- inferred from user behavior;
- denser but noisier --- absence $\neq$ dislike;
- examples: clicks, purchases, watch time, skips.

### Approaches to recommendation

**Content-based filtering**:
- recommend items similar to those the user liked;
- requires item features (genre, author, keywords);
- no interaction with other users.

**Collaborative filtering**:
- exploit patterns across all users and items;
- no item features required;
- "users who liked $X$ also liked $Y$".

**Hybrid methods**: combine content-based and collaborative filtering.

## Collaborative Filtering

### Key idea

~~~center
\textit{``Tell me who your friends are, and I will tell you what you like.''}
~~~

\vspace{4mm}

**Assumptions**:
- users with similar past behavior will have similar future preferences;
- items that appealed to similar users will appeal to the target user.

**Two families**:
- **user-based**: find users similar to the target user, aggregate their ratings;
- **item-based**: find items similar to the target item, aggregate ratings of those items.

### User-based collaborative filtering

**Algorithm**:
1. compute similarity $\mathrm{sim}(u, v)$ between target user $u$ and every other user $v$;
2. select $k$ nearest neighbors $\mathcal{N}(u)$;
3. predict the rating as a similarity-weighted average.

~~~equation*
\hat{R}_{ui} = \bar{r}_u + \frac{\displaystyle\sum_{v \in \mathcal{N}(u)} \mathrm{sim}(u, v)\,(R_{vi} - \bar{r}_v)}{\displaystyle\sum_{v \in \mathcal{N}(u)} |\mathrm{sim}(u, v)|}
~~~

where $\bar{r}_u = \frac{1}{|\Omega_u|} \sum_{j \in \Omega_u} R_{uj}$ is the mean rating of user $u$.

### Similarity measures

**Cosine similarity** (treats missing as zero):
~~~equation*
\mathrm{sim}_{\cos}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\|\,\|\mathbf{r}_v\|}
~~~

**Pearson correlation** (uses co-rated items only):
~~~equation*
\mathrm{sim}_{\mathrm{P}}(u, v) =
  \frac{\displaystyle\sum_{i \in \Omega_{uv}} (R_{ui} - \bar{r}_u)(R_{vi} - \bar{r}_v)}
       {\sqrt{\displaystyle\sum_{i \in \Omega_{uv}} (R_{ui} - \bar{r}_u)^2}\;
        \sqrt{\displaystyle\sum_{i \in \Omega_{uv}} (R_{vi} - \bar{r}_v)^2}}
~~~

where $\Omega_{uv}$ = items rated by both $u$ and $v$.

### Item-based collaborative filtering

**Algorithm**:
1. compute similarity $\mathrm{sim}(i, j)$ between target item $i$ and every other item $j$;
2. select $k$ nearest neighbor items $\mathcal{N}(i)$;
3. predict the rating using the user's ratings of neighboring items.

~~~equation*
\hat{R}_{ui} = \frac{\displaystyle\sum_{j \in \mathcal{N}(i)} \mathrm{sim}(i, j)\,R_{uj}}
                    {\displaystyle\sum_{j \in \mathcal{N}(i)} |\mathrm{sim}(i, j)|}
~~~

**Key difference**: similarities are between items, not users.

### User-based vs item-based

~~~center
\begin{tabular}{lll}
\toprule
\textbf{Property} & \textbf{User-based} & \textbf{Item-based} \\
\midrule
Similarity space  & users ($m \times m$)   & items ($n \times n$) \\
Stability         & low (user tastes change) & high (catalog is stable) \\
Sparsity handling & struggles              & more robust \\
Explanation       & ``users like you liked\dots'' & ``similar to item $X$'' \\
\bottomrule
\end{tabular}
~~~

\vspace{3mm}
**In practice**: item-based CF is often preferred --- item similarities are more stable and can be precomputed offline.

### Limitations of neighborhood methods

**Sparsity**:
- most users rate very few items;
- little overlap $\Omega_{uv}$ between user pairs.

**Scalability**:
- all-pairs similarity: $O(m^2)$ or $O(n^2)$;
- prohibitive for millions of users/items.

**Cold start**:
- new users or items have no ratings;
- no similarity can be computed.

**Quality ceiling**:
- neighborhood heuristics are not globally consistent;
- matrix factorization methods typically outperform them.

## Matrix Factorization

### Latent factor models

**Intuition**: ratings are driven by a small number of hidden factors.

- each user $u$ is described by a latent vector $\mathbf{p}_u \in \mathbb{R}^k$;
- each item $i$ is described by a latent vector $\mathbf{q}_i \in \mathbb{R}^k$;
- predicted rating:
~~~equation*
\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i
~~~

**Examples of latent factors** (movies):
- seriousness vs.\ humor;
- action-oriented vs.\ dialogue-driven;
- mainstream vs.\ arthouse.

### Matrix factorization

Approximate the full rating matrix as a product of two low-rank matrices:

~~~equation*
R \approx P Q^\top, \qquad P \in \mathbb{R}^{m \times k},\quad Q \in \mathbb{R}^{n \times k}
~~~

where $k \ll \min(m, n)$ is the number of latent factors.

- - -

~~~center
\begin{tabular}{ccccc}
$R$ & $\approx$ & $P$ & $\cdot$ & $Q^\top$ \\[2mm]
$m \times n$ & & $m \times k$ & & $k \times n$ \\
\end{tabular}
~~~

### Loss function

**Objective** (minimize reconstruction error on observed entries only):

~~~equation*
\min_{P,\,Q}\; \sum_{(u,i)\in\Omega} \left(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\right)^2
  + \lambda\!\left(\|P\|_F^2 + \|Q\|_F^2\right)
~~~

- sum restricted to **observed** entries in $\Omega$;
- $\ell_2$ regularization with strength $\lambda > 0$ prevents overfitting;
- **non-convex** jointly in $(P, Q)$.

### Bias terms

~~~equation*
\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i
~~~

where:
- $\mu$ --- global mean rating;
- $b_u$ --- user bias (some users rate higher overall);
- $b_i$ --- item bias (some items are rated higher overall).

Note: $\mathbf{p}'_u = (1, b_u, \mathbf{p}_u)^\top$, $\mathbf{q}'_i = (b_i, 1, \mathbf{q}_i)^\top$ --- a special case of the dot product model.

### Why not just use SVD?

**Truncated SVD** minimizes:
~~~equation*
\min_{\mathrm{rank}\,k}\; \|R - \hat{R}\|_F^2
~~~

over **all** entries --- but $R$ has missing values!

**Naive workaround**: impute missing entries with 0 or $\bar{r}$.

**Problems**:
- imputing zeros biases toward low ratings;
- imputing the mean ignores structure;
- the resulting factorization fits the imputed values, not the true ones.

**Correct approach**: optimize only over observed entries $\Omega$.

## Alternating Least Squares

### ALS: motivation

The combined loss is non-convex in $(P, Q)$:

~~~equation*
\mathcal{L}(P, Q) = \sum_{(u,i)\in\Omega}(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2
  + \lambda\!\left(\|P\|_F^2 + \|Q\|_F^2\right)
~~~

But it is **convex** in $P$ alone when $Q$ is fixed, and vice versa.

**ALS strategy**: alternate between closed-form updates of $P$ and $Q$.

### ALS: update for user factors

Fix $Q$, minimize over each $\mathbf{p}_u$ independently.

Let $\Omega_u = \{i : (u,i) \in \Omega\}$ be the items rated by user $u$:

~~~equation*
\mathbf{p}_u = \left(Q_u^\top Q_u + \lambda I\right)^{-1} Q_u^\top \mathbf{r}_u
~~~

where:
- $Q_u \in \mathbb{R}^{|\Omega_u| \times k}$ --- rows of $Q$ for items in $\Omega_u$;
- $\mathbf{r}_u \in \mathbb{R}^{|\Omega_u|}$ --- observed ratings of user $u$.

This is a **ridge regression** problem per user!

### ALS: update for item factors

Fix $P$, minimize over each $\mathbf{q}_i$ independently.

Let $\Omega_i = \{u : (u,i) \in \Omega\}$ be the users who rated item $i$:

~~~equation*
\mathbf{q}_i = \left(P_i^\top P_i + \lambda I\right)^{-1} P_i^\top \mathbf{r}_i
~~~

where:
- $P_i \in \mathbb{R}^{|\Omega_i| \times k}$ --- rows of $P$ for users in $\Omega_i$;
- $\mathbf{r}_i \in \mathbb{R}^{|\Omega_i|}$ --- observed ratings for item $i$.

Symmetric to the user update --- also ridge regression!

### ALS algorithm

~~~framed
\begin{spacing}{1.5}
\begin{algorithmic}[1]
  \State Initialize $P$, $Q$ randomly (e.g.\ $\sim \mathcal{N}(0,\,\sigma^2)$)
  \Repeat
    \For{each user $u = 1,\dots,m$}
      \State $\mathbf{p}_u \leftarrow \left(Q_u^\top Q_u + \lambda I\right)^{-1} Q_u^\top \mathbf{r}_u$
    \EndFor
    \For{each item $i = 1,\dots,n$}
      \State $\mathbf{q}_i \leftarrow \left(P_i^\top P_i + \lambda I\right)^{-1} P_i^\top \mathbf{r}_i$
    \EndFor
  \Until{convergence}
\end{algorithmic}
\end{spacing}
~~~

### ALS: Parallelism:
- all user updates are independent $\Rightarrow$ trivially parallelizable;
- all item updates are independent $\Rightarrow$ trivially parallelizable;
- widely used in distributed settings (Apache Spark MLlib).

**Complexity per iteration**:
~~~equation*
O\!\left(|\Omega|\,k + (m + n)\,k^3\right)
~~~

### Weighted ALS for implicit feedback

**Implicit feedback** (Hu, Koren, Volinsky 2008):
- let $R_{ui} = 1$ if user $u$ interacted with item $i$, else $0$;
- absence of interaction $\neq$ dislike --- only lower confidence.

**Weighted objective** (sum over **all** pairs):
~~~equation*
\min_{P,Q}\;\sum_{u=1}^{m}\sum_{i=1}^{n} c_{ui}\,(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2
  + \lambda\!\left(\|P\|_F^2 + \|Q\|_F^2\right)
~~~

**Confidence**:
~~~equation*
c_{ui} = 1 + \alpha\;\mathrm{count}_{ui}
~~~
where $\mathrm{count}_{ui}$ is the number of interactions (plays, clicks, \dots).

### Poisson factorization for implicit feedback

**Model** (Gopalan, Hofman, Blei 2015):
~~~equation*
\mathrm{count}_{ui} \sim \mathrm{Poisson}\!\left(\mathbf{p}_u^\top \mathbf{q}_i\right), \quad \mathbf{p}_u, \mathbf{q}_i \geq 0
~~~

Negative Log-Likelihood:
~~~equation*
L(P, Q) = -\sum_{u,i} \left[ \mathbf{p}_u^\top \mathbf{q}_i - \mathrm{count}_{ui} \log(\mathbf{p}_u^\top \mathbf{q}_i) \right]
~~~

- models **count data** (plays, clicks, views);
- Missing entries contribute via the $-\mathbf{p}_u^\top \mathbf{q}_i$ term;

### Poisson factorization: log-linear parameterization

Instead of constraining $\mathbf{p}_u, \mathbf{q}_i \geq 0$, parameterize the rate directly:
~~~equation*
\lambda_{ui} = \exp\!\left(\mathbf{p}_u^\top \mathbf{q}_i\right), \quad \mathbf{p}_u, \mathbf{q}_i \in \mathbb{R}^k
~~~

The log-likelihood becomes:
~~~equation*
\sum_{u,i} \left[ \mathrm{count}_{ui} \cdot \mathbf{p}_u^\top \mathbf{q}_i - e^{\mathbf{p}_u^\top \mathbf{q}_i} \right]
~~~

- $\mathbf{p}_u, \mathbf{q}_i$ are unconstrained --- standard SGD applies directly;
- Gradients: $\nabla_{\mathbf{p}_u} = \sum_i (\mathrm{count}_{ui} - \lambda_{ui})\,\mathbf{q}_i$,
- ... symmetrically for $\mathbf{q}_i$.

### Evaluation: explicit feedback

**Root Mean Squared Error**:
~~~equation*
\mathrm{RMSE} = \sqrt{\frac{1}{|\Omega_{\mathrm{test}}|}
  \sum_{(u,i)\in\Omega_{\mathrm{test}}} \left(R_{ui} - \hat{R}_{ui}\right)^2}
~~~

**Mean Absolute Error**:
~~~equation*
\mathrm{MAE} = \frac{1}{|\Omega_{\mathrm{test}}|}
  \sum_{(u,i)\in\Omega_{\mathrm{test}}} |R_{ui} - \hat{R}_{ui}|
~~~

### Evaluation: ranking metrics

For implicit feedback or top-$N$ recommendation:

**Precision\@$N$**:
~~~equation*
\mathrm{Precision@}N = \frac{|\text{relevant items in top-}N|}{N}
~~~

**NDCG\@$N$** (Normalized Discounted Cumulative Gain):
~~~equation*
\mathrm{DCG@}N = \sum_{j=1}^{N} \frac{\mathrm{rel}_j}{\log_2(j+1)}, \qquad
\mathrm{NDCG@}N = \frac{\mathrm{DCG@}N}{\mathrm{IDCG@}N}
~~~

where $\mathrm{IDCG}$ is the ideal (perfect) ranking.

### Practical considerations

**Choosing rank $k$**:
- too small: underfits, misses structure;
- too large: overfits, slow to train;
- typical range: $k \in \{10, 50, 200\}$.

**Choosing $\lambda$**:
- cross-validation on held-out ratings;
- typical range: $\lambda \in \{0.01, 0.1, 1.0\}$.

**Initialization**:
- small random values: $P, Q \sim \mathcal{N}(0, k^{-1/2})$;
- avoids large initial predictions.

### Cold start problem

**New user** (no interaction history):
- cannot compute $\mathbf{p}_u$ directly;
- solutions: ask for initial ratings (onboarding), use demographic features, or start from the global mean vector.

**New item** (no ratings yet):
- cannot compute $\mathbf{q}_i$ directly;
- solutions: initialize from item content features, use a separate content-based model.

**Fundamental limitation** of pure collaborative filtering.

### Summary

Recommendation systems:
- Collaborative Filtering:
  - item-based;
  - user-based;
- Matrix Factorization;
- Alternating Least Squares.

\vspace{3mm}
ALS-based matrix factorization dominates large-scale production systems (Netflix Prize winner, Spotify, etc.).