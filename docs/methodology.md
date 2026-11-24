# Mathematical Methodology: Unsupervised Weather Observation Anomaly Detection

**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  
**Project Title:** Faulty Weather-Station Observation Detection Using Anomaly Detection Methods  

---

## 1. Feature Representation & Scaling

Given a set of $N$ daily observations across $P$ numerical meteorological variables, each observation vector is denoted:
$$x_i = [TMAX_i, TMIN_i, TEMP\_RANGE_i, PRCP_i, SNOW_i, SNWD_i, \Delta TMAX_i, \Delta TMIN_i, \sin(DOY_i), \cos(DOY_i)]^T \in \mathbb{R}^{10}$$

### Feature Normalization
Because precipitation and snow depth span different orders of magnitude from temperature, standard z-score normalization is applied:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
where $\mu_j = \frac{1}{N}\sum_{i=1}^N x_{ij}$ and $\sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_{ij} - \mu_j)^2}$. Standard scaling preserves outlier tails in multivariate space without artificial truncation.

---

## 2. Isolation Forest Formulation

Isolation Forest (Liu, Ting & Zhou, 2008) operates on the principle that anomalous instances are "few and different", making them susceptible to rapid isolation under recursive random partitioning.

### Algorithm Mechanics:
1. An ensemble of $T$ isolation trees (iTrees) is constructed.
2. At each node of an iTree, an attribute $q$ is randomly selected from available features, and a split value $p$ is selected uniformly at random between $\min(q)$ and $\max(q)$.
3. The recursive partitioning terminates when:
   - The tree reaches a maximum height limit $h_{max} = \lceil \log_2(\psi) \rceil$ (sub-sampling size $\psi$).
   - The node contains $|X| \le 1$ observations.
   - All data points at the node have identical attribute values.

### Anomaly Score Derivation:
The path length $h(x)$ is the number of edges traversed from the root node to the terminating external leaf. The average path length of unsuccessful searches in a Binary Search Tree (BST) serves as the normalizing factor:
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
where $0.5772156649$ is Euler's constant.

The anomaly score $s(x, n)$ for instance $x$ over an ensemble of $T$ isolation trees is:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
- As $\mathbb{E}[h(x)] \to 0 \implies s \to 1$: instance is isolated in very few splits, indicating a strong anomaly.
- As $\mathbb{E}[h(x)] \to c(n) \implies s \to 0.5$: instance exhibits typical average depth, indicating normal inlier behavior.
- As $\mathbb{E}[h(x)] \to n - 1 \implies s \to 0$: instance requires deep splits, indicating dense central clustering.

---

## 3. Local Outlier Factor (LOF) Formulation

Local Outlier Factor (Breunig et al., 2000) measures local density deviation relative to an observation's $k$-nearest neighbors, effectively identifying outliers in regions of varying density.

### Key Mathematical Steps:
1. **$k$-distance of object $p$, $d_k(p)$:**
   The Euclidean distance $d(p, o)$ between $p$ and an object $o \in D$ such that at least $k$ objects are at distance $\le d(p, o)$ and at most $k-1$ objects are at distance $< d(p, o)$.

2. **$k$-distance neighborhood of $p$, $N_k(p)$:**
   $$N_k(p) = \{q \in D \setminus \{p\} \mid d(p, q) \le d_k(p)\}$$

3. **Reachability Distance:**
   The reachability distance of object $p$ with respect to object $o$ prevents statistical instability for dense clusters:
   $$\text{reach-dist}_k(p, o) = \max\{d_k(o), d(p, o)\}$$

4. **Local Reachability Density ($lrd_k$):**
   The inverse of the average reachability distance over the $k$-nearest neighbors:
   $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$

5. **Local Outlier Factor ($LOF_k$):**
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
   - $\text{LOF} \approx 1.0 \implies$ point has density comparable to neighbors (normal).
   - $\text{LOF} \gg 1.0 \implies$ point has substantially lower density than neighbors (local outlier).

---

## 4. Principal Component Analysis (PCA) Formulation

PCA performs an orthogonal linear basis transformation to project standardized features $Z \in \mathbb{R}^{N \times P}$ onto $M = 2$ principal component axes:
$$\Sigma = \frac{1}{N} Z^T Z = V \Lambda V^T$$
where $V = [v_1, v_2, \dots, v_P]$ is the orthonormal eigenvector matrix, and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_P)$ contains sorted eigenvalues ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_P$).

The 2D projection is obtained via:
$$Z_{\text{pca}} = Z \cdot [v_1, v_2] \in \mathbb{R}^{N \times 2}$$

The proportion of total dataset variance explained by principal component $m$ is:
$$\text{EVR}_m = \frac{\lambda_m}{\sum_{j=1}^P \lambda_j}$$
For our 10-dimensional meteorological feature space:
$$\text{EVR}_1 = 27.96\%, \quad \text{EVR}_2 = 16.08\%, \quad \text{Cumulative (2D)} = 44.05\%$$

---

## 5. Comparative Evaluation Metrics

In unsupervised settings without artificial ground truth labels, model consistency is evaluated via:

### Jaccard Similarity Index:
Let $S_{\text{IF}}$ and $S_{\text{LOF}}$ denote the sets of observation indices flagged as anomalies:
$$J(S_{\text{IF}}, S_{\text{LOF}}) = \frac{|S_{\text{IF}} \cap S_{\text{LOF}}|}{|S_{\text{IF}} \cup S_{\text{LOF}}|}$$

### Total Agreement Rate:
$$\text{Agreement} = \frac{|\{i \mid y_i^{\text{IF}} = y_i^{\text{LOF}}\}|}{N} \times 100\%$$
Across our baseline configuration ($c=0.03, k=20$), the models achieve **95.33% total agreement**, demonstrating high background consensus while providing complementary detection capabilities on the distribution margins.

