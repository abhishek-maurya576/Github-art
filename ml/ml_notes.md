## BCA120: Machine Learning Techniques
### Unit 1: Introduction to Machine Learning

---

### 1.1 Introduction: Basic Definitions

Let's begin by defining what Machine Learning is and why it's such a transformative field.

*   **What is Machine Learning (ML)?**
    *   ML is a subset of Artificial Intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions with minimal human intervention.
    *   Arthur Samuel (1959) defined it as: "the field of study that gives computers the ability to learn without being explicitly programmed."
    *   More formally, a computer program is said to *learn* from *experience E* with respect to some *task T* and performance measure *P*, if its performance on T, as measured by P, improves with experience E (Tom Mitchell, 1997).

*   **Why do we use Machine Learning?**
    *   **Tasks where explicit programming is difficult:** Recognizing images, speech, translating languages.
    *   **Finding hidden patterns:** Market basket analysis, scientific discovery.
    *   **Prediction:** Stock prices, weather, customer churn.
    *   **Personalization:** Recommending products, content.
    *   **Automation:** Fraud detection, quality control.

*   **Key Components:**
    *   **Data:** The raw material. ML algorithms learn from data.
    *   **Algorithm:** The process or set of rules used to learn from data.
    *   **Model:** The output of the learning process. It's the learned function that maps input data to output predictions or insights.

*   **Important Definitions:**
    *   **Instance / Sample:** A single data point or observation (e.g., details of one customer, one image).
    *   **Feature / Attribute:** A characteristic or property of an instance (e.g., customer age, pixel value). Also called a predictor or independent variable.
    *   **Label / Target:** The output variable we are trying to predict or understand (e.g., whether a customer will churn, the object in the image). Also called dependent variable or response.
    *   **Training Data:** The dataset used to train the machine learning model. The algorithm learns patterns from this data.
    *   **Testing Data:** An independent dataset used to evaluate the performance of the trained model on unseen data. This gives an estimate of how well the model will generalize.

---

### 1.2 Types of Machine Learning

Machine learning problems are broadly categorized based on the nature of the data and the task.

*   **Supervised Learning:**
    *   **Definition:** Learning from *labeled* data, where each training instance has a known input and a corresponding desired output (label).
    *   **Goal:** To learn a mapping function from inputs to outputs so we can predict the output for new, unseen inputs.
    *   **Tasks:**
        *   **Classification:** Predicting a *discrete* class label (e.g., spam/not spam, cat/dog, healthy/diseased).
        *   **Regression:** Predicting a *continuous* numerical value (e.g., house price, temperature, stock value).
    *   **Examples:** Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines (SVMs), Neural Networks.

*   **Unsupervised Learning:**
    *   **Definition:** Learning from *unlabeled* data, where there are no predefined output labels.
    *   **Goal:** To find hidden patterns, structures, or relationships within the data.
    *   **Tasks:**
        *   **Clustering:** Grouping similar data instances together (e.g., customer segmentation).
        *   **Dimensionality Reduction:** Reducing the number of features while preserving important information (e.g., compressing data, visualizing high-dimensional data).
        *   **Association Rule Mining:** Discovering relationships between variables (e.g., "customers who buy bread also tend to buy milk").
    *   **Examples:** K-Means Clustering, Principal Component Analysis (PCA), Self-Organizing Maps (SOM).

*   **Ensemble Learning:**
    *   **Definition:** Combining predictions from multiple individual models to improve overall performance and robustness.
    *   **Goal:** Often achieves better accuracy than any single model alone. Reduces variance and bias.
    *   **Techniques:**
        *   **Bagging:** Training multiple models independently on different subsets of the data (e.g., Random Forests). Averages or votes on predictions.
        *   **Boosting:** Training models sequentially, where each new model focuses on correcting the errors made by the previous ones (e.g., AdaBoost, Gradient Boosting, XGBoost).
    *   **Examples:** Random Forests, Gradient Boosting Machines.

*   *(Note: Reinforcement Learning is another major type, where an agent learns by taking actions in an environment and receiving rewards or penalties. While listed later in the syllabus, the main types introduced early are typically Supervised, Unsupervised, and Ensemble).*

---

### 1.3 Hypothesis Space and Inductive Bias

These are fundamental concepts explaining how an algorithm "learns" and generalizes.

*   **Hypothesis (h):** A candidate function or model that attempts to map inputs to outputs. It's a potential explanation for the data.
*   **Hypothesis Space (H):** The set of *all possible* hypotheses that a learning algorithm can potentially learn or search within. For example, if we're doing linear regression, our hypothesis space might be all possible lines (y = mx + c). If we're learning decision trees, it's all possible decision trees.
*   **Learning as Search:** The learning process can be viewed as searching through the hypothesis space H to find the best hypothesis `h` that fits the training data well and is expected to generalize well to unseen data.

*   **Inductive Bias:**
    *   **Definition:** The set of assumptions or preferences that a learning algorithm uses to select one hypothesis over others from the hypothesis space, especially when multiple hypotheses fit the training data equally well.
    *   **Why is it necessary?** Without some bias, the algorithm would only memorize the training data (like fitting a complex curve through every point), which performs poorly on unseen data. Bias helps the algorithm generalize. It guides the search for a hypothesis that is likely to work well beyond the training set.
    *   **Types of Bias:**
        *   **Restriction Bias:** Limits the hypothesis space itself (e.g., assuming the relationship *must* be linear).
        *   **Preference Bias:** Orders hypotheses within the space, preferring some over others (e.g., preferring shorter decision trees, preferring smoother functions).
    *   **Example:**
        *   Linear Regression has a strong restriction bias: it *only* considers linear functions.
        *   Decision Tree algorithms like ID3 or C4.5 have a preference bias for smaller trees (Occam's Razor - simpler explanations are preferred), achieved by preferring splits that maximize information gain early on.

*   **Relationship:** The inductive bias determines which part of the vast hypothesis space H the algorithm explores and how it chooses a specific hypothesis `h` from the hypotheses consistent with the training data. A good bias aligns well with the underlying patterns in the problem.

---

### 1.4 Evaluation and Cross-Validation

Once we have trained a model, how do we know if it's any good? We need to evaluate its performance.

*   **Evaluation:**
    *   **Goal:** To estimate how well our trained model will perform on *unseen* data. This is crucial for understanding its real-world utility and comparing different models.
    *   **Process:** We use a separate **test set** that the model has *not* seen during training.
    *   **Metrics:** The choice of metric depends on the task:
        *   **For Classification:** Accuracy (proportion of correct predictions), Precision, Recall, F1-score, AUC (Area Under the ROC Curve).
        *   **For Regression:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared.
    *   **Importance:** Evaluating on the training set gives an *optimistic* and often misleading estimate of performance because the model has already seen and potentially memorized this data.

*   **Cross-Validation:**
    *   **Problem with simple Train/Test Split:** If we only have a limited dataset, splitting it once into train and test sets might result in a test set that isn't representative, leading to an unreliable performance estimate. Also, it reduces the amount of data available for training.
    *   **Solution:** Cross-validation is a technique to get a more robust estimate of model performance and make better use of the available data.
    *   **K-Fold Cross-Validation (Most Common):**
        1.  Divide the entire dataset into `K` equally sized subsets or "folds".
        2.  Repeat the training and evaluation process `K` times.
        3.  In each iteration `i` (from 1 to K):
            *   Use fold `i` as the **test set**.
            *   Use the remaining `K-1` folds combined as the **training set**.
            *   Train the model on the training set.
            *   Evaluate the trained model on the test set and record the performance metric.
        4.  Calculate the average of the `K` performance scores recorded. This average provides a more reliable estimate of the model's performance on unseen data.
        5.  *Common choice for K is 5 or 10.*
        6.  **Leave-One-Out Cross-Validation (LOOCV):** A special case where K = N (the number of instances in the dataset). Each instance is used as a test set once, and the remaining N-1 instances are used for training. Computationally expensive for large datasets.

*   **Purpose of Cross-Validation:**
    *   Get a more reliable estimate of the model's true performance on unseen data.
    *   Helps detect if the model is overly sensitive to the specific train/test split.
    *   Used for model selection (comparing different algorithms) and hyperparameter tuning (finding the best settings for an algorithm).

---

### 1.5 Linear Regression

The first specific model type we'll look at, and a classic example of supervised *regression*.

*   **Goal:** To predict a *continuous* target variable `y` based on one or more input features `X`.
*   **Assumption:** There is a *linear relationship* between the features and the target variable.
*   **Model Form:**
    *   **Simple Linear Regression (one feature):** `y = w0 + w1 * x1 + ε`
        *   `y`: target variable
        *   `x1`: input feature
        *   `w0`: intercept (bias term)
        *   `w1`: coefficient (weight) for feature x1
        *   `ε`: error term (noise)
    *   **Multiple Linear Regression (multiple features):** `y = w0 + w1*x1 + w2*x2 + ... + wn*xn + ε`
        *   In vector form: `y = W^T * X + ε` (where W includes w0 and X includes a constant 1) or often written as `y = W * X + b`.
*   **Learning:** Finding the best values for the weights (`w` or `W`) and the intercept (`w0` or `b`) that minimize the difference between the predicted values (`ŷ`) and the actual target values (`y`) on the training data.
*   **Loss Function:** The most common loss function for linear regression is the **Mean Squared Error (MSE)** or Sum of Squared Errors (SSE). The goal is to minimize this function.
    *   `MSE = (1/N) * Σ (yi - ŷi)^2` (sum of squared differences between actual and predicted values)
*   **Finding the Weights:** Can be found using:
    *   **Ordinary Least Squares (OLS):** A closed-form mathematical solution.
    *   **Gradient Descent:** An iterative optimization algorithm that gradually adjusts weights to minimize the loss function.

*   **Example:** Predicting house price based on size (simple LR), number of bedrooms, location, etc. (multiple LR).

---

### 1.6 Decision Trees

Another fundamental model, used for both supervised *classification* and *regression*.

*   **Goal:** To create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
*   **Structure:** A flowchart-like tree structure.
    *   **Internal Nodes:** Represent a test on a feature (e.g., "Is temperature > 25?").
    *   **Branches:** Represent the outcomes of the test (e.g., "Yes" or "No").
    *   **Leaf Nodes:** Represent the final prediction (a class label for classification, or a numerical value for regression).
*   **Learning (Building the Tree):**
    *   The tree is built recursively from the root node.
    *   At each node, the algorithm selects the feature and the split point that best divides the data into more "pure" subsets with respect to the target variable.
    *   **Purity Measures (for Classification):**
        *   **Gini Impurity:** Measures how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. A split aims to minimize Gini impurity in the resulting subsets.
        *   **Entropy / Information Gain:** Entropy measures the disorder or randomness in a set of labels. Information Gain is the reduction in entropy achieved by splitting on a particular feature. A split aims to maximize information gain.
    *   The process stops when a stopping criterion is met (e.g., nodes are sufficiently pure, a maximum depth is reached, or there are too few instances in a node).
*   **Making a Prediction:** Traverse the tree from the root, following the branches based on the feature tests, until a leaf node is reached. The value in the leaf node is the prediction.

*   **Advantages:**
    *   Easy to understand and interpret ("white box" model).
    *   Can handle both numerical and categorical data.
    *   Requires little data preprocessing (no feature scaling needed).
*   **Disadvantages:**
    *   Can easily **overfit** the training data (create very complex trees that don't generalize well).
    *   Can be unstable (small changes in data can lead to a very different tree).
    *   Greedy approach to learning (makes the best split at the current step, not necessarily the best for the overall tree).

---

### 1.7 Overfitting

A major challenge in Machine Learning that we must be aware of and learn to combat.

*   **Definition:** Occurs when a model learns the training data *too well*, including the noise and random fluctuations, to the point where it performs poorly on new, unseen data. The model has essentially memorized the training examples rather than learning the underlying general pattern.
*   **Analogy:** Like a student who memorizes answers for a test without understanding the concepts. They will do well on that specific test (training data) but fail a test with different questions (test data).
*   **Causes:**
    *   Model is too complex relative to the amount of training data (e.g., a very deep decision tree, a polynomial regression model of high degree).
    *   Insufficient training data.
    *   Data contains a lot of noise.
*   **Symptoms:** Very high performance on the training set, but significantly lower performance on the test set.
*   **Consequences:** The model fails to generalize. It won't be useful in real-world applications where it encounters new data.

*   **How to Deal with Overfitting:**
    *   **Get More Training Data:** More data helps the model distinguish between signal and noise.
    *   **Use a Simpler Model:** Choose an algorithm that is less prone to complexity, or use a simpler version of the same algorithm (e.g., shallow decision tree, linear model instead of polynomial).
    *   **Regularization:** Add a penalty term to the loss function that discourages complex models (e.g., L1 or L2 regularization for linear models, upcoming in later units).
    *   **Cross-Validation:** Use cross-validation during model selection and hyperparameter tuning to get a better estimate of generalization performance and choose models/settings that perform well on unseen folds.
    *   **Feature Selection/Extraction:** Reduce the number of features used, focusing only on the most informative ones.
    *   **Pruning (for trees):** Remove branches from a decision tree that contribute little to classification/regression power but might be fitting noise.

---

### 1.8 Design a Learning System: Perspectives and Issues

Designing and deploying a machine learning system involves several steps and considerations.

*   **Steps in Designing/Implementing an ML System:**
    1.  **Problem Definition:** Clearly understand the task, the goal, and the desired outcome. What are we trying to predict or discover?
    2.  **Data Collection:** Gather relevant data. The quality and quantity of data are crucial.
    3.  **Data Preprocessing / Cleaning:** Handle missing values, outliers, incorrect data formats, etc.
    4.  **Feature Engineering:** Select, transform, or create features from the raw data that are most informative for the model. This is often the most critical step!
    5.  **Model Selection:** Choose an appropriate machine learning algorithm(s) based on the problem type, data characteristics, and desired performance/interpretability.
    6.  **Training:** Train the chosen model(s) on the prepared training data.
    7.  **Evaluation:** Evaluate the trained model's performance using appropriate metrics on the test set or through cross-validation. Tune hyperparameters.
    8.  **Deployment:** Integrate the trained model into the application or system.
    9.  **Monitoring and Maintenance:** Continuously monitor the model's performance in the real world and retrain or update it as needed, as data patterns can change over time (concept drift).

*   **Perspectives:** Different viewpoints or frameworks for thinking about and building ML systems.
    *   **Statistical/Probabilistic:** Focusing on probability distributions and statistical inference (e.g., Naive Bayes, Gaussian Mixture Models).
    *   **Geometric/Algebraic:** Focusing on distances, hyperplanes, and vector spaces (e.g., SVMs, K-Means).
    *   **Logical/Symbolic:** Focusing on rules and logical structures (e.g., Decision Trees, Rule-based systems).
    *   **Connectionist:** Inspired by biological neural networks (e.g., Neural Networks, Deep Learning).

*   **Key Issues in Machine Learning:**
    *   **Data Quality:** "Garbage in, garbage out." Biased, noisy, or insufficient data will lead to poor models.
    *   **Feature Selection/Engineering:** Choosing or creating the right features is paramount.
    *   **Model Selection:** Choosing the right algorithm for the task and data.
    *   **Overfitting vs. Underfitting:** Finding the right model complexity. Underfitting occurs when the model is too simple to capture the underlying patterns.
    *   **Bias and Fairness:** ML models can perpetuate and amplify biases present in the training data, leading to unfair or discriminatory outcomes.
    *   **Interpretability vs. Performance:** Complex models (like deep neural networks) often achieve high performance but are hard to understand ("black box"). Simpler models (like decision trees, linear models) are more interpretable but may have lower performance.
    *   **Computational Cost:** Training large models on massive datasets requires significant computing resources and time.
    *   **Scalability:** How well does the algorithm and system handle increasing amounts of data?
    *   **Security and Privacy:** Protecting sensitive data used for training and preventing adversarial attacks on models.

---

### 1.9 Feature Engineering in Machine Learning

Revisiting this crucial step in the design process.

*   **Definition:** The process of using domain knowledge to select, transform, and create relevant features from raw data to improve model performance. It's about making the data *understandable* and *informative* for the learning algorithm.
*   **Why is it Important?**
    *   Most ML algorithms work best with well-structured, numeric features.
    *   Raw data is often messy, incomplete, or in unsuitable formats (e.g., text, dates).
    *   Creating new features can highlight relationships or patterns that the raw data doesn't explicitly show.
    *   A good set of features can make even a simple model perform very well. Poor features can make even complex models struggle.

*   **Common Techniques:**
    *   **Handling Missing Values:** Imputation (filling with mean, median, mode, or using model predictions), deletion.
    *   **Encoding Categorical Variables:** Converting text categories into numerical formats (e.g., One-Hot Encoding, Label Encoding).
    *   **Scaling / Normalization:** Rescaling numerical features to a standard range (e.g., 0-1) or standard deviation (mean 0, std dev 1). Important for algorithms sensitive to feature scales (like SVMs, K-Means, Neural Networks).
    *   **Creating New Features:**
        *   Extracting information from dates (day of week, month, year).
        *   Combining existing features (e.g., calculating BMI from height and weight).
        *   Creating polynomial features (e.g., `x^2`, `x^3`).
        *   Creating interaction terms (e.g., `x1 * x2`).
    *   **Handling Text Data:** Tokenization, stemming, TF-IDF, Word Embeddings.
    *   **Handling Image Data:** Pixel manipulations, edge detection, using pre-trained networks for feature extraction.

*   **It's an iterative process:** Feature engineering often involves experimentation and domain expertise.

---

### 1.10 Unit 1 Summary

In this introductory unit, we laid the groundwork for our study of Machine Learning.

*   We defined ML as enabling computers to learn from data without explicit programming, focusing on improving performance on a task with experience.
*   We explored the main **types of learning**: **Supervised** (learning from labeled data for classification/regression), **Unsupervised** (finding patterns in unlabeled data for clustering/dimensionality reduction), and **Ensemble** (combining models).
*   We discussed the theoretical underpinnings: the **Hypothesis Space** (the set of possible models) and **Inductive Bias** (the algorithm's assumptions that guide the search within that space and enable generalization).
*   We looked at the practical steps involved in **Designing a Learning System**, highlighting common **Perspectives and Issues** like data quality, bias, overfitting, and interpretability.
*   We emphasized the importance of **Feature Engineering** – the art of transforming raw data into informative features for the model.
*   We introduced two foundational models as examples of supervised learning: **Linear Regression** (for predicting continuous values with linear relationships) and **Decision Trees** (for learning rule-based predictions, prone to overfitting).
*   Crucially, we defined **Overfitting** (memorizing training data instead of generalizing) and introduced essential techniques for assessing a model's true performance and combating overfitting: **Evaluation on a separate test set** and **Cross-Validation**.

This unit provides the essential context and basic tools to understand and approach the more complex techniques we will cover in subsequent units.

***
---
# ****Unit 2****

---

**Unit 2: Unsupervised Learning, Instance-Based Methods, and Probability**

### 2.1 Unsupervised Learning: Deeper Dive

*   **Recap:** Unsupervised learning algorithms work with **unlabeled data**. There is no target variable or correct output provided during training.
*   **Goal:** To explore the data and find hidden patterns, structures, or relationships.
    *   Discover clusters (grouping similar data).
    *   Reduce dimensionality (simplifying data representation).
    *   Find association rules (identifying relationships between items).
    *   Detect anomalies (finding unusual data points).

### 2.2 Clustering Algorithms

Clustering is the task of partitioning the dataset into groups (clusters) such that instances within the same cluster are more similar to each other than to instances in other clusters.

*   **Definition:** The process of grouping a set of data instances into clusters based on some measure of similarity or distance.

*   **Key Idea:** We define similarity (or distance) between data points, and the algorithm tries to find groups that are internally cohesive and externally separated. Common distance metrics include Euclidean distance, Manhattan distance, etc.

*   **2.2.1 K-Means Algorithm**
    *   **Type:** Partitioning Clustering (Part of Unsupervised Learning).
    *   **Goal:** To partition `n` data points into `k` clusters, where `k` is a pre-specified number.
    *   **Core Idea:** Iteratively assign data points to the nearest cluster center (centroid) and then update the cluster centers based on the mean of the assigned points.
    *   **Algorithm Steps:**
        1.  **Initialization:** Choose the number of clusters, `k`. Randomly select `k` data points from the dataset as the initial cluster centroids.
        2.  **Assignment Step:** Assign each data point to the nearest centroid. This forms `k` clusters.
        3.  **Update Step:** Recalculate the centroid for each cluster. The new centroid is the mean of all data points assigned to that cluster.
        4.  **Repeat:** Repeat steps 2 and 3 until the centroids no longer change significantly (convergence) or a maximum number of iterations is reached.
    *   **Objective:** K-Means aims to minimize the sum of squared distances between data points and their assigned cluster centroid (within-cluster sum of squares).
    *   **Choosing K:** A challenge is determining the optimal `k`. Methods include the Elbow Method or Silhouette Score analysis.
    *   **Pros:** Relatively simple to understand and implement, computationally efficient for large datasets.
    *   **Cons:** Requires specifying `k` beforehand, sensitive to initial centroid placement (can converge to local optima), assumes clusters are spherical and of similar size, sensitive to outliers.

*   **2.2.2 Adaptive Hierarchical Clustering**
    *   **Type:** Hierarchical Clustering (Part of Unsupervised Learning). The term "Adaptive" might refer to specific implementations or the iterative nature of building/splitting, but the core concept is building a hierarchy.
    *   **Goal:** To build a hierarchy of clusters, rather than a simple partitioning. This hierarchy is often represented as a tree-like structure called a **dendrogram**.
    *   **Two Main Approaches:**
        *   **Agglomerative (Bottom-Up):** Starts with each data point as its own cluster. Then, iteratively merges the two nearest clusters until only one cluster (containing all data points) remains.
        *   **Divisive (Top-Down):** Starts with all data points in one large cluster. Then, recursively splits the clusters into smaller ones until each data point is in its own cluster.
    *   **Merging/Splitting Criteria:** Based on distance/similarity between clusters. Common linkage methods for agglomerative clustering:
        *   **Single Linkage:** Distance between two clusters is the minimum distance between any point in one cluster and any point in the other.
        *   **Complete Linkage:** Distance between two clusters is the maximum distance between any point in one cluster and any point in the other.
        *   **Average Linkage:** Distance is the average distance between all pairs of points across the two clusters.
        *   **Ward's Method:** Merges clusters to minimize the increase in total within-cluster variance.
    *   **Interpretation:** The dendrogram shows the sequence of merges (agglomerative) or splits (divisive). You can choose the number of clusters by cutting the dendrogram at a certain height.
    *   **Pros:** Does not require specifying `k` beforehand (you decide the number of clusters by cutting the dendrogram), provides a hierarchical view of the data structure, can reveal relationships between clusters.
    *   **Cons:** Computationally more expensive than K-Means, especially for large datasets (often O(n³)), once a merge/split decision is made, it cannot be undone, sensitive to noise and outliers.

*   **2.2.3 Gaussian Mixture Model (GMM)**
    *   **Type:** Model-Based Clustering (Part of Unsupervised Learning), relies on probability.
    *   **Goal:** To model the probability distribution of the data as a mixture of multiple Gaussian (normal) distributions. Each Gaussian component is assumed to represent a cluster.
    *   **Core Idea:** Assumes the data points are generated from a mixture of `k` Gaussian distributions, each with its own mean and covariance matrix. The algorithm estimates the parameters (mean, covariance, and weight) of each Gaussian component.
    *   **How it works (Simplified):** Uses an iterative algorithm called **Expectation-Maximization (EM)**.
        1.  **Initialization:** Initialize parameters (means, covariances, weights) for `k` Gaussian components.
        2.  **Expectation (E) Step:** Calculate the probability that each data point belongs to each Gaussian component (cluster), based on the current parameter estimates. This gives us soft assignments (a point can belong to a cluster with a certain probability).
        3.  **Maximization (M) Step:** Update the parameters (mean, covariance, weight) of each Gaussian component based on the soft assignments calculated in the E-step.
        4.  **Repeat:** Repeat E and M steps until parameters converge.
    *   **Output:** For each data point, GMM provides the probability of belonging to each cluster. A point is typically assigned to the cluster it has the highest probability of belonging to.
    *   **Choosing K:** Similar to K-Means, choosing `k` is necessary, often done using information criteria like BIC or AIC.
    *   **Pros:** More flexible than K-Means (can model clusters with different shapes and sizes), provides probabilistic cluster assignments (soft clustering).
    *   **Cons:** Assumes data is generated from Gaussian distributions, sensitive to initialization, computationally more expensive than K-Means, sensitive to outliers, requires specifying `k`.

### 2.3 Other Unsupervised/Related Techniques

*   **2.3.1 Vector Quantization**
    *   **Concept:** A technique used for data compression and signal processing. It works by grouping a large number of vectors (data points) into a smaller number of clusters (prototypes or codewords).
    *   **Relation to K-Means:** K-Means is a common algorithm used to perform Vector Quantization. The centroids learned by K-Means act as the "codewords" or representative vectors. Each data point is then represented by the index of the centroid it's closest to.
    *   **Application:** Image compression (grouping similar pixel blocks), audio compression, feature extraction.

*   **2.3.2 Self Organizing Feature Map (SOM)**
    *   **Type:** A type of artificial neural network (specifically, an unsupervised competitive learning network). Also known as Kohonen map.
    *   **Goal:** To produce a low-dimensional (typically 2D) discrete map of the input space, preserving the topological properties of the input data. It essentially projects high-dimensional data onto a grid.
    *   **Core Idea:** Has an input layer and an output layer arranged in a grid (the map). Neurons in the output layer are competitive. When an input vector is presented, neurons on the map compete, and the "winning" neuron (the one whose weights are most similar to the input) and its neighbors have their weights updated to be more like the input vector. Over iterations, neighboring neurons on the map learn to represent similar input vectors.
    *   **Result:** The grid of neurons forms a map where nearby neurons represent similar data points in the input space. This can be used for visualization and exploratory data analysis.
    *   **Pros:** Useful for visualizing high-dimensional data, preserves topological relationships, can handle non-linear relationships.
    *   **Cons:** Sensitive to initialization and parameter choices, fixed map size/topology needs to be chosen beforehand, can be harder to interpret than other methods.

### 2.4 Instance-Based Learning and Nearest Neighbour Methods

*   **Definition (Instance-Based Learning):** These algorithms don't explicitly build a model from the training data upfront. Instead, they store the entire training dataset. When a new instance needs a prediction, they compare it to the stored instances and make a decision based on the properties of the most similar instances.
    *   Sometimes called "lazy learning" because computation is deferred until prediction time.

*   **2.4.1 Nearest Neighbour Methods (K-Nearest Neighbors - KNN)**
    *   **Type:** Instance-Based, Non-Parametric. *Although placed in the Unsupervised unit in the syllabus, KNN is predominantly used for **Supervised Learning** (Classification and Regression).* We will cover it here as an instance-based method.
    *   **Goal:** To predict the label (class for classification, value for regression) of a new data point based on the labels of its `k` nearest neighbors in the training data.
    *   **Algorithm Steps (for prediction on a new instance):**
        1.  Choose the number of neighbors, `k`.
        2.  Calculate the distance (e.g., Euclidean distance) between the new instance and *every* instance in the training dataset.
        3.  Identify the `k` training instances with the smallest distances (the k-nearest neighbors).
        4.  **For Classification:** The new instance's class is predicted by a majority vote among the k-nearest neighbors' classes.
        5.  **For Regression:** The new instance's value is predicted by taking the average (or weighted average) of the k-nearest neighbors' values.
    *   **Key Factors:** Choice of `k` and choice of distance metric.
    *   **Pros:** Simple to understand and implement, no training phase (computation is all at prediction time), can model complex decision boundaries.
    *   **Cons:** Prediction can be slow for large datasets (requires calculating distances to all training points), performance is sensitive to the choice of `k` and distance metric, sensitive to irrelevant features and the scale of features, requires storing the entire training dataset.

### 2.5 Probability in Machine Learning

Probability theory provides a mathematical framework for dealing with uncertainty, which is fundamental in machine learning. Many ML algorithms are based on probabilistic principles.

*   **Basic Concepts:**
    *   **Probability:** A measure of the likelihood of an event occurring.
    *   **Random Variable:** A variable whose value is subject to variations due to chance.
    *   **Probability Distribution:** Describes the probabilities of all possible outcomes for a random variable (e.g., Gaussian distribution, Bernoulli distribution).
    *   **Conditional Probability:** P(A|B) - The probability of event A occurring given that event B has already occurred.
    *   **Joint Probability:** P(A, B) - The probability of both event A and event B occurring.
    *   **Independent Events:** P(A, B) = P(A) * P(B).
    *   **Bayes' Theorem:** Relates conditional probabilities. P(A|B) = [P(B|A) * P(A)] / P(B). This is fundamental to Bayesian approaches in ML.

*   **Role in ML:**
    *   **Modeling Uncertainty:** Many ML models (like GMMs, Bayesian Networks, Logistic Regression) explicitly model the uncertainty in data or predictions using probability distributions.
    *   **Making Decisions:** Probability can be used to make optimal decisions under uncertainty (e.g., classifying a point based on the probability it belongs to each class).
    *   **Probabilistic Models:** Algorithms like GMMs model the data's underlying probability distribution. This allows not just clustering but also generating new data points that resemble the training data.
    *   **Evaluation:** Metrics like likelihood or probability scores can be used to evaluate models.

*(Note: The syllabus mentions "Bayes learning" here. While Bayes' Theorem is key, specific algorithms like Naive Bayes Classifier are supervised methods typically covered elsewhere. The main link in this unit is how probability underpins models like Gaussian Mixture Models).*

### 2.6 Feature Reduction (Brief Mention)

*(As explicit techniques like PCA, LDA are in Unit 5, this section is brief).*

*   **Concept:** The process of reducing the number of features (dimensions) in a dataset.
*   **Why:** To reduce computational complexity, reduce storage space, potentially improve model performance by removing noisy or irrelevant features, and help with visualization.
*   **Methods:**
    *   **Feature Selection:** Choosing a subset of the original features.
    *   **Feature Extraction:** Creating new, lower-dimensional features from the original features (e.g., PCA, LLE - covered in Unit 5).
*   **Relevance in Unsupervised Learning:** Dimensionality reduction is a major task within unsupervised learning, helping to simplify data representation or enable visualization.

### 2.7 Summary of Unit 2

Unit 2 explored several key areas:

*   We focused on **Unsupervised Learning**, where the goal is to find patterns in unlabeled data.
*   We covered **Clustering** techniques:
    *   **K-Means:** Partitioning data into `k` clusters based on centroids, using an iterative assignment and update process. Simple but sensitive to initialization and assumes spherical clusters.
    *   **Hierarchical Clustering:** Building a tree-like hierarchy (dendrogram) of clusters (Agglomerative or Divisive). Provides a view of structure but is computationally intensive.
    *   **Gaussian Mixture Models (GMM):** Modeling data as a mixture of Gaussian distributions using EM, providing probabilistic cluster assignments and handling non-spherical clusters. Requires specifying `k` and assumes underlying Gaussian distributions.
*   We looked at **Vector Quantization** (often implemented with K-Means for data compression) and **Self Organizing Feature Maps (SOM)** (a neural network for projecting data onto a topological map for visualization).
*   We introduced **Instance-Based Learning**, where the training data itself is the model. The prime example covered was **K-Nearest Neighbors (KNN)**, primarily a supervised algorithm for classification/regression, but fundamentally instance-based.
*   We briefly touched upon the role of **Probability** in machine learning, explaining basic concepts and its relevance in models like GMMs for handling uncertainty and modeling data distributions.
*   We mentioned **Feature Reduction** as a goal, noting that specific techniques will be covered later.

This unit expanded our toolkit beyond just learning from labeled examples, showing how ML can find hidden structures and relationships within data.

---
---
---
# **Unit 3** 

---
**Unit 3: Discriminative Models, Kernels, and Neural Networks**

This unit covers several fundamental supervised learning algorithms, shifting towards models that often learn decision boundaries explicitly or implicitly. We'll start with Logistic Regression, move to Support Vector Machines and the concept of kernels, and finally introduce the basics of Neural Networks, leading into deep learning.

### 3.1 Logistic Regression

*   **Type:** Supervised Learning (primarily **Classification**).
*   **Despite the name "regression",** Logistic Regression is a classification algorithm used for predicting categorical outcomes, most commonly binary classification (two classes, e.g., Yes/No, 0/1). It can be extended for multi-class classification.
*   **Goal:** To model the probability that an instance belongs to a particular class, given its features. It then uses these probabilities to make a classification decision.
*   **How it Works:**
    1.  It calculates a linear combination of the input features and their weights, similar to Linear Regression: `z = w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ` (or `z = wᵀx + b`).
    2.  Instead of outputting `z` directly, it passes `z` through a **Sigmoid Function (or Logistic Function)**.
*   **Sigmoid Function (σ):**
    *   **Definition:** σ(z) = 1 / (1 + e⁻ᶻ)
    *   **Properties:**
        *   Takes any real-valued number `z` as input.
        *   Outputs a value between 0 and 1.
        *   As `z` approaches positive infinity, σ(z) approaches 1.
        *   As `z` approaches negative infinity, σ(z) approaches 0.
        *   When `z = 0`, σ(z) = 0.5.
    *   **Role:** The sigmoid function transforms the linear output `z` into a probability estimate, P(Y=1 | X), which is the probability that the instance belongs to the positive class (class 1), given the features X.
*   **Decision Boundary:** A threshold (typically 0.5) is applied to the output probability. If P(Y=1 | X) ≥ 0.5, the instance is classified as class 1; otherwise, it's class 0. The decision boundary is the line/hyperplane where `z = 0` (since σ(0) = 0.5).
*   **Learning:** The algorithm learns the optimal weights (`w`) and bias (`b`) that minimize a cost function.
*   **Cost Function (Loss Function):** **Cross-Entropy Loss (or Log Loss)** is typically used for classification problems like Logistic Regression. It penalizes the model more when it is confident about a wrong prediction. The goal is to maximize the likelihood of the training data given the model parameters.
*   **Optimization:** Gradient Descent or other optimization algorithms are used to find the weights and bias that minimize the Cross-Entropy Loss.
*   **Pros:** Simple, interpretable (weights indicate feature importance), computationally efficient, outputs probabilities.
*   **Cons:** Assumes a linear decision boundary, sensitive to outliers, assumes features are not highly correlated.

### 3.2 Support Vector Machine (SVM)

*   **Type:** Supervised Learning (primarily **Classification**, can be extended to Regression - SVR).
*   **Goal:** To find the optimal hyperplane that separates the data points of different classes in the feature space. For binary classification.
*   **Core Idea:** Instead of just finding *any* separating hyperplane (if one exists), SVM finds the hyperplane that has the largest margin.
*   **Hyperplane:** In a p-dimensional feature space, a hyperplane is a flat subspace of dimension p-1. In 2D, it's a line; in 3D, it's a plane.
*   **Margin:** The distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this distance (the "maximum margin classifier").
*   **Support Vectors:** The data points from the training set that are closest to the hyperplane and lie on the edge of the margin. These points are crucial because they "support" the hyperplane; if they were removed, the optimal hyperplane might change. All other training points are irrelevant once the hyperplane is found.
*   **Hard Margin vs. Soft Margin:**
    *   **Hard Margin SVM:** Assumes the data is perfectly linearly separable. Finds a hyperplane that separates *all* training points without any errors. Very sensitive to outliers.
    *   **Soft Margin SVM:** Allows for some misclassifications or points within the margin to handle non-linearly separable data and noise. It introduces a cost parameter (C) that controls the trade-off between maximizing the margin and minimizing misclassification errors. A smaller C allows more errors/smaller margin (more regularization), a larger C aims for fewer errors/larger margin (less regularization).
*   **Learning:** Involves solving a convex optimization problem to find the weights (`w`) and bias (`b`) of the hyperplane that maximize the margin (subject to constraints).

### 3.3 Kernel Function and Kernel SVM

*   **Problem:** Standard (Linear) SVM only works well if the data is linearly separable (or nearly so with soft margin). What about non-linearly separable data?
*   **Solution Idea:** Map the data into a higher-dimensional feature space where it *is* linearly separable. Then, find a linear hyperplane in that higher dimension. This hyperplane, when mapped back to the original feature space, corresponds to a non-linear decision boundary.
*   **Example:** Consider 1D data where positive points are around 0 and negative points are further away. Not linearly separable in 1D. Map to 2D using feature `x²`. The points now form a parabola, which is linearly separable with a horizontal line `y = constant`.
*   **The "Kernel Trick":** Explicitly mapping data to a very high-dimensional space can be computationally expensive or even impossible. The kernel trick avoids this explicit mapping.
    *   **Concept:** SVM's optimization problem involves calculating dot products of feature vectors (`xᵢ ⋅ xⱼ`). A kernel function `K(xᵢ, xⱼ)` is a function that calculates this dot product *as if* the data were in a higher dimension, *without* actually performing the mapping and calculating the high-dimensional vectors explicitly. `K(xᵢ, xⱼ) = φ(xᵢ) ⋅ φ(xⱼ)`, where `φ` is the mapping function to the higher dimension.
*   **Kernel SVM:** SVM that uses a kernel function. It allows SVM to learn non-linear decision boundaries in the original feature space by implicitly working in a higher (or infinite) dimensional space.
*   **Common Kernel Functions:**
    *   **Linear Kernel:** `K(xᵢ, xⱼ) = xᵢ ⋅ xⱼ` (Equivalent to standard linear SVM).
    *   **Polynomial Kernel:** `K(xᵢ, xⱼ) = (γ * xᵢ ⋅ xⱼ + r)ᵈ` (d is the degree).
    *   **Radial Basis Function (RBF) Kernel (or Gaussian Kernel):** `K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)`. This is one of the most popular kernels; `γ` is a parameter that influences the shape of the decision boundary.
*   **Pros (Kernel SVM):** Highly effective on non-linearly separable data, powerful and versatile.
*   **Cons:** Can be computationally expensive for large datasets, choosing the right kernel and its hyperparameters (like `γ` or `C`) can be tricky, less interpretable than linear models.

### 3.4 Neural Network (Introduction)

Neural Networks (NNs) are a class of algorithms inspired by the structure and function of the human brain's neurons.

*   **Basic Unit: The Neuron (or Perceptron)**
    *   **Analogy:** A biological neuron receives signals through dendrites, processes them in the cell body, and sends a signal through the axon if the input exceeds a threshold.
    *   **Artificial Neuron Model:**
        *   Receives inputs (x₁, x₂, ..., xₙ), typically feature values.
        *   Each input has an associated weight (w₁, w₂, ..., wₙ).
        *   Calculates a weighted sum of inputs: `z = w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ` (where w₀ is the bias, x₀=1 implicitly).
        *   Passes the sum through an **Activation Function (f)** to produce the output: `output = f(z)`. The activation function introduces non-linearity.

*   **3.4.1 Perceptron (Single Layer)**
    *   **Type:** Simplest type of feedforward neural network (single layer of output neurons, no hidden layers). Supervised Learning.
    *   **Structure:** Input layer connected directly to an output layer of one or more neurons.
    *   **Activation Function:** Originally used a Step Function (outputs 1 if sum > threshold, 0 otherwise).
    *   **Learning (Perceptron Learning Rule):** An algorithm to adjust weights. If the prediction for a training instance is wrong, update weights to reduce the error. The update rule is simple: `wᵢ = wᵢ + learning_rate * (actual_output - predicted_output) * xᵢ`.
    *   **Capability:** Can only learn **linearly separable patterns**. It cannot solve problems like the XOR problem.
    *   **Historical Significance:** Important foundation but limited in practice for complex problems.

*   **3.4.2 Multilayer Network (Multilayer Perceptron - MLP)**
    *   **Type:** Feedforward Neural Network with one or more hidden layers between the input and output layers. Supervised Learning.
    *   **Structure:**
        *   **Input Layer:** Receives the features.
        *   **Hidden Layer(s):** Intermediate layers of neurons. Each neuron in a hidden layer receives input from the previous layer, applies weights and bias, and passes the result through an activation function. These layers allow the network to learn complex, non-linear relationships.
        *   **Output Layer:** Produces the final prediction(s).
    *   **Connectivity:** Typically, neurons are fully connected between adjacent layers (dense layers).
    *   **Activation Functions:** Often use non-linear functions like ReLU (Rectified Linear Unit), Sigmoid, or Tanh in hidden layers to enable learning non-linear patterns. The output layer activation depends on the task (e.g., Sigmoid for binary classification, Softmax for multi-class classification, Linear for regression).
    *   **Capability:** Can learn **arbitrarily complex non-linear patterns** (given enough hidden neurons and layers - Universal Approximation Theorem).

*   **3.4.3 Back Propagation**
    *   **Definition:** The primary algorithm used to train Multilayer Perceptrons (and other feedforward networks) by calculating the gradient of the loss function with respect to the network's weights and biases. This gradient is then used by an optimization algorithm (like Gradient Descent) to update the parameters.
    *   **Core Idea:** An efficient way to compute the gradients needed for optimization. It's based on the chain rule of calculus.
    *   **Process (Two Passes):**
        1.  **Forward Pass:**
            *   Input data is fed into the network.
            *   Calculations are performed layer by layer (weighted sum + activation) to produce an output prediction.
            *   The loss (error) between the prediction and the actual target is calculated.
        2.  **Backward Pass:**
            *   The calculated error is propagated backward through the network, starting from the output layer.
            *   Using the chain rule, the contribution of each weight and bias to the total error is calculated (i.e., the gradient).
            *   These gradients indicate the direction and magnitude to change each parameter to reduce the error.
    *   **Weight Update:** An optimization algorithm (e.g., Stochastic Gradient Descent - SGD, Adam, RMSprop) uses the computed gradients to update the weights and biases, iteratively minimizing the loss function over the training data.
    *   **Significance:** Backpropagation made training multi-layer neural networks practical, leading to their widespread use.

*   **3.4.4 Introduction to Deep Neural Network (DNN)**
    *   **Definition:** A Neural Network with **multiple hidden layers** (typically more than one or two). "Deep" refers to the depth of the network structure (number of layers).
    *   **Why Deep?** Deep networks can learn hierarchical representations of data. Lower layers might learn simple features (like edges in images), while higher layers combine these into more complex features (like shapes, objects). This hierarchical learning allows them to model highly complex patterns.
    *   **Rise of Deep Learning:** Became feasible due to:
        *   Availability of large datasets.
        *   Significant increases in computational power (especially GPUs).
        *   Algorithmic improvements (e.g., better activation functions like ReLU, better optimization algorithms, regularization techniques).
    *   **Examples of Deep Architectures:** Convolutional Neural Networks (CNNs) for images, Recurrent Neural Networks (RNNs) for sequential data, Transformers.
    *   **Impact:** Revolutionized areas like computer vision, natural language processing, speech recognition, etc.

### 3.5 Summary of Unit 3

Unit 3 introduced several key supervised learning models:

*   **Logistic Regression:** A linear model for **classification** that uses the sigmoid function to output probability estimates, trained using Cross-Entropy loss. Useful for binary classification and provides interpretable weights.
*   **Support Vector Machines (SVM):** Aims to find the maximum-margin hyperplane for classification. The concept of **Support Vectors** (critical training points near the margin) is central.
*   **Kernel SVM:** Extends SVM to handle non-linearly separable data using the **Kernel Trick**, which allows implicitly mapping data to a higher dimension using **Kernel Functions** (like RBF) without the computational cost of explicit mapping.
*   **Neural Networks:** Models inspired by the brain structure.
    *   The basic unit is the **Neuron/Perceptron**.
    *   A single-layer **Perceptron** is limited to linear problems.
    *   **Multilayer Perceptrons (MLPs)** with hidden layers and non-linear activation functions can learn complex non-linear patterns.
    *   **Back Propagation** is the essential algorithm for training MLPs, using gradient descent to minimize error by propagating error signals backward.
    *   **Deep Neural Networks (DNNs)** are MLPs with many hidden layers, capable of learning powerful hierarchical representations, driving the recent success of deep learning.

These models represent a significant step towards handling more complex data and decision boundaries compared to simpler linear methods.

---
---
---
---
# **Unit 4** 
---


**Unit 4: Learning with Trees and Ensemble Methods**

This unit expands on decision trees and introduces powerful techniques for combining multiple models to improve performance, known as ensemble learning. We also touch again on the role of probability and basic statistics, which are fundamental to many ML approaches.

### 4.1 Learning with Trees: Decision Trees

*   **Recap:** Decision Trees are a type of supervised learning algorithm that makes predictions by following a tree-like structure of decisions. They can be used for both Classification and Regression tasks.
*   **Structure:**
    *   **Root Node:** The starting point, represents the entire dataset.
    *   **Internal Nodes:** Represent a test on a specific feature (attribute). The data is split based on the outcome of the test.
    *   **Branches:** Represent the possible outcomes of the test at an internal node.
    *   **Leaf Nodes (Terminal Nodes):** Represent the final decision or prediction (a class label for classification, a numerical value for regression).

*   **How Prediction Works:** To classify or predict for a new instance, you traverse the tree from the root node down to a leaf node by following the path dictated by the instance's feature values. The value in the leaf node is the prediction.

*   **4.1.1 Constructing Decision Trees**
    *   Decision trees are typically built using a greedy, top-down, recursive approach.
    *   **Core Idea:** At each node, select the best feature to split the data such that the child nodes are as "pure" as possible (meaning instances in each child node are mostly of the same class for classification, or have similar values for regression).
    *   **Algorithm Steps:**
        1.  Start with the entire dataset at the root node.
        2.  Evaluate potential splits on all features. For each feature and possible split point, measure how well it separates the data.
        3.  Select the feature and split point that results in the "best" separation according to a predefined **splitting criterion**.
        4.  Create child nodes based on the chosen split.
        5.  Recursively repeat steps 2-4 for each child node until a stopping condition is met (e.g., nodes are sufficiently pure, maximum depth is reached, minimum number of instances in a node is reached).
    *   **Stopping Conditions:** Prevent the tree from growing too deep and overfitting the training data.

*   **4.1.2 Splitting Criteria**
    *   Used to quantify the "purity" of a node or the "information gain" from a split. The goal is to minimize impurity or maximize gain.
    *   **For Classification:**
        *   **Gini Impurity:** Measures how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. A Gini impurity of 0 means the node is perfectly pure (all instances belong to the same class). A split is chosen to minimize the weighted average Gini impurity of the child nodes.
        *   **Entropy (Information Gain):** Entropy measures the randomness or disorder in a set of labels. Information Gain is the reduction in entropy achieved by splitting the data on a particular feature. A split is chosen to maximize information gain.
    *   **For Regression:**
        *   **Variance Reduction:** Measures how much the variance of the target variable is reduced by splitting the data. A split is chosen to minimize the weighted average variance in the child nodes. The prediction in a leaf node is typically the average of the target values of instances in that node.

*   **4.1.3 Classification and Regression Trees (CART)**
    *   CART is a specific implementation of the decision tree algorithm.
    *   It uses **Gini Impurity** as the splitting criterion for classification trees and **Mean Squared Error (equivalent to variance reduction)** for regression trees.
    *   CART produces **binary trees**, meaning each internal node has exactly two branches (a "yes" or "no" split, e.g., `feature <= value` or `feature > value`).

*   **Decision Tree Pros:**
    *   Easy to understand and interpret (can be visualized).
    *   Handle both numerical and categorical data.
    *   Can capture non-linear relationships.
    *   Require little data preprocessing (e.g., no feature scaling needed).
*   **Decision Tree Cons:**
    *   Prone to **overfitting**, especially if the tree is deep.
    *   Small changes in data can lead to a very different tree structure (instability).
    *   Can create biased trees if some classes dominate.
    *   Finding the *optimal* tree is NP-hard; greedy algorithms are used.

### 4.2 Ensemble Learning

*   **Recap:** Combining multiple individual models (base learners) to improve overall performance and robustness.
*   **Goal:** Leverage the "wisdom of crowds". By aggregating predictions from diverse models, ensemble methods often achieve higher accuracy and better generalization than any single model.
*   **Why it Works:** Ensemble methods can help reduce:
    *   **Variance:** By averaging or voting, the impact of individual models' errors or sensitivity to specific training data subsets is reduced (common in Bagging).
    *   **Bias:** By combining multiple weak learners, a complex concept can be learned that a single weak learner cannot (common in Boosting).
    *   **Overfitting:** By combining models trained differently or on different data, the ensemble is less likely to have memorized the noise in the training data in the same way (especially Bagging).

*   **4.2.1 Bagging (Bootstrap Aggregating)**
    *   **Concept:** Train multiple independent base learners on different **bootstrap samples** of the training data.
    *   **Bootstrap Sample:** A random sample of the training data taken **with replacement**. A bootstrap sample has the same size as the original training set but will likely contain duplicates and omit some original instances.
    *   **Procedure:**
        1. Create `N` bootstrap samples from the original training data.
        2. Train a base learner (often a Decision Tree) on each of the `N` bootstrap samples independently.
        3. **Combine Predictions:**
            *   For **Classification:** Use majority voting among the `N` base learners' predictions.
            *   For **Regression:** Use the average of the `N` base learners' predictions.
    *   **Key Feature:** Base learners are trained in parallel and independently.
    *   **Example:** **Random Forests**. A Random Forest is a bagging ensemble specifically using Decision Trees as base learners. Additionally, when building each tree, it only considers a random subset of features at each split point. This further decorrelates the trees, reducing variance.

*   **4.2.2 Boosting**
    *   **Concept:** Train multiple base learners sequentially. Each new learner is trained to focus on the instances that the previous learners misclassified or predicted poorly.
    *   **Procedure:**
        1. Train an initial base learner (often a "weak" learner, slightly better than random).
        2. Evaluate the learner and identify misclassified instances (for classification) or instances with high errors (for regression).
        3. Adjust the weights of the training instances: increase the weight of misclassified/high-error instances, decrease the weight of correctly classified/low-error instances.
        4. Train the next base learner on the dataset with the adjusted instance weights.
        5. Repeat steps 2-4 for a fixed number of iterations or until convergence.
        6. **Combine Predictions:** Predictions are combined using a weighted sum, where learners that performed better on the weighted data are given higher importance.
    *   **Key Feature:** Base learners are trained sequentially, with each learning from the mistakes of the previous ones.
    *   **Examples:**
        *   **AdaBoost (Adaptive Boosting):** One of the earliest boosting algorithms. Adjusts instance weights and assigns weights to the base learners based on their accuracy.
        *   **Gradient Boosting:** A more general framework where each new learner is trained to predict the *gradient* of the loss function with respect to the ensemble's current prediction. Popular implementations include Gradient Boosting Machines (GBM), XGBoost, LightGBM, CatBoost.

*   **4.2.3 Different ways to Combine Classifiers**
    *   Beyond bagging (voting/averaging) and boosting (weighted voting/summing), other combination strategies exist:
        *   **Simple Majority Vote:** Each classifier gets one vote.
        *   **Weighted Voting:** Classifiers are assigned weights based on their performance (e.g., test accuracy), and their votes are weighted accordingly.
        *   **Averaging (for Regression or Probabilities):** Simply average the predictions or predicted probabilities.
        *   **Weighted Averaging:** Average predictions/probabilities using performance-based weights.
        *   **Stacking (Stacked Generalization):** Train multiple base learners. Then, train a "meta-learner" (a higher-level model) using the predictions of the base learners as input features to make the final prediction.

### 4.3 Probability and Learning, Data into Probabilities, Basic Statistics, Gaussian Mixture Models, Nearest Neighbour Methods

This section reinforces concepts introduced earlier, highlighting their relevance in the broader ML context, potentially as preparation for later units or to solidify understanding.

*   **4.3.1 Probability and Learning:**
    *   Many ML algorithms have a probabilistic foundation (e.g., Naive Bayes, Logistic Regression, GMMs, probabilistic graphical models).
    *   Probability helps model uncertainty in data and predictions.
    *   It allows framing learning as finding parameters that maximize the likelihood of observing the training data (Maximum Likelihood Estimation - MLE) or that are most probable given the data and prior beliefs (Maximum A Posteriori - MAP estimation).

*   **4.3.2 Data into Probabilities:**
    *   Converting raw data or model outputs into probability estimates.
    *   Example: Logistic Regression outputs probabilities via the sigmoid function.
    *   Example: For discrete data, estimating probabilities by counting frequencies (e.g., P(Spam | "Viagra") = Count(Spam & "Viagra") / Count("Viagra")).
    *   Example: For continuous data, estimating probability density functions (PDFs) using parametric models (like Gaussian in GMMs) or non-parametric methods (like Kernel Density Estimation).

*   **4.3.3 Basic Statistics:**
    *   Essential for understanding data properties and many ML algorithms.
    *   **Descriptive Statistics:** Mean, Median, Mode (measures of central tendency); Variance, Standard Deviation, Range (measures of dispersion); Skewness, Kurtosis (measures of shape).
    *   **Inferential Statistics:** Concepts like hypothesis testing, confidence intervals (used for model evaluation and comparison).
    *   **Relationships:** Covariance, Correlation (measures of linear relationship between variables - crucial for methods like PCA in Unit 5).
    *   Many ML algorithms implicitly or explicitly rely on minimizing error metrics that are statistical in nature (e.g., MSE relates to variance).

*   **4.3.4 Gaussian Mixture Models (GMM)**
    *   **Revisit from Unit 2:** GMM is a model-based clustering algorithm.
    *   **Probabilistic Nature:** It models the data's probability distribution as a weighted sum of multiple Gaussian distributions.
    *   Each Gaussian component represents a potential cluster, defined by its mean (center), covariance matrix (shape and orientation), and weight (proportion of data belonging to it).
    *   Training uses the Expectation-Maximization (EM) algorithm to find the parameters that best fit the data's distribution.
    *   Output is a probabilistic assignment of each data point to each cluster.

*   **4.3.5 Nearest Neighbour Methods**
    *   **Revisit from Unit 2/3:** K-Nearest Neighbors (KNN) is an instance-based algorithm (typically supervised).
    *   **Core Idea:** Classifies/predicts a new point based on the majority class/average value of its `k` nearest training data points.
    *   **Connection to Probability/Statistics (Subtle):** While KNN itself isn't explicitly probabilistic, the class probability for a new point can be estimated based on the proportion of its `k` neighbors belonging to each class. It implicitly uses distance metrics which relate to spatial statistics. It's a non-parametric method, making no assumptions about the underlying data distribution.

### 4.4 Summary of Unit 4

Unit 4 deepened our understanding of tree-based models and introduced the powerful concept of combining models:

*   We revisited **Decision Trees**, covering their structure (nodes, branches, leaves), how they are **Constructed** using greedy algorithms and **Splitting Criteria** like Gini Impurity, Entropy (for Classification Trees), and Variance Reduction (for Regression Trees). We specifically mentioned the **CART** algorithm.
*   We introduced **Ensemble Learning** as a method to improve performance and robustness by combining multiple base learners.
*   We explored two main ensemble techniques:
    *   **Bagging (Bootstrap Aggregating):** Training models on parallel bootstrap samples and combining via voting/averaging (e.g., **Random Forests**). Reduces variance.
    *   **Boosting:** Training models sequentially, focusing on previous errors, and combining via weighted summing (e.g., AdaBoost, Gradient Boosting variants). Reduces bias, turns weak learners into strong ones.
*   We briefly reinforced the fundamental importance of **Probability** and **Basic Statistics** in understanding data and various ML models.
*   We revisited **Gaussian Mixture Models** as a probabilistic model for clustering.
*   We touched upon **Nearest Neighbour Methods** (like KNN) again as an instance-based approach, noting its non-parametric nature.

This unit provided tools for building both single, interpretable tree models and highly performant, robust ensemble models. The emphasis on combining models is a major theme in practical machine learning.

---
---
---
---

# **unit 5**

---


**Unit 5: Dimensionality Reduction, Evolutionary Models, and Reinforcement Learning**

This unit introduces techniques to handle high-dimensional data, explores a different paradigm of learning inspired by evolution, and provides an overview of learning by interacting with an environment.

### 5.1 Dimensionality Reduction

*   **Definition:** The process of reducing the number of random variables (features) under consideration.
*   **Goal:** To represent the data in a lower-dimensional space while retaining as much of the important information (variance, structure) as possible.
*   **Why Reduce Dimensionality?**
    *   **Curse of Dimensionality:** In high dimensions, data becomes sparse, distances between points become less intuitive, and models struggle to find meaningful patterns.
    *   **Computational Efficiency:** Algorithms run faster on lower-dimensional data.
    *   **Storage Space:** Less memory/disk space required.
    *   **Visualization:** Difficult to visualize data in more than 3 dimensions; reducing to 2D or 3D enables plotting.
    *   **Noise Reduction:** Removing irrelevant or noisy features can sometimes improve model performance.

*   **Two main approaches:**
    *   **Feature Selection:** Choosing a subset of the original features.
    *   **Feature Extraction:** Creating a new set of features (components) that are combinations or transformations of the original features. The new features live in a lower-dimensional space.

*   **Key Techniques (Feature Extraction):**

    *   **5.1.1 Principal Component Analysis (PCA)**
        *   **Type:** Unsupervised (typically, though variants exist). Linear Feature Extraction.
        *   **Goal:** To find a set of orthogonal (uncorrelated) axes (Principal Components) that capture the maximum variance in the data.
        *   **Core Idea:** Projects the data onto a lower-dimensional subspace such that the variance of the projected data is maximized. The first principal component accounts for the most variance, the second the next most (orthogonal to the first), and so on.
        *   **How it works (Simplified):** Involves calculating the covariance matrix of the data, finding its eigenvectors and eigenvalues. The eigenvectors represent the directions (principal components), and the eigenvalues represent the magnitude of variance along those directions. You select the top `k` eigenvectors (those with the largest eigenvalues) to form the new k-dimensional subspace.
        *   **Pros:** Widely used, effective for linear dimensionality reduction, projects onto uncorrelated components.
        *   **Cons:** Assumes linear relationships, sensitive to feature scaling, the new components might not be easily interpretable (they are linear combinations of original features).

    *   **5.1.2 Linear Discriminant Analysis (LDA)**
        *   **Type:** Supervised Linear Feature Extraction.
        *   **Goal:** To find a set of linear axes that maximize the separation *between* classes while minimizing the variance *within* each class.
        *   **Core Idea:** Unlike PCA which focuses on maximizing overall variance regardless of class labels, LDA uses class labels to find a projection that is best for *discrimination*.
        *   **How it works (Simplified):** Calculates within-class scatter and between-class scatter matrices. Finds the projection axes that maximize the ratio of between-class scatter to within-class scatter.
        *   **Pros:** Specifically designed for classification tasks, can provide good class separation.
        *   **Cons:** Assumes normal distribution within classes and equal covariance matrices, sensitive to outliers, limited by the number of classes (can project to at most C-1 dimensions, where C is the number of classes).

    *   **5.1.3 Factor Analysis (FA)**
        *   **Type:** Unsupervised (typically). Statistical model-based Feature Extraction.
        *   **Goal:** To model the observed variables as linear combinations of a smaller number of unobserved (latent) factors plus unique error terms. Assumes the shared variance among observed variables is due to these underlying factors.
        *   **Core Idea:** Attempts to explain the correlations between observed variables through a smaller set of common factors.
        *   **Relation to PCA:** Both reduce dimensionality and involve concepts related to covariance. However, PCA is a data transformation method aiming to capture variance, while FA is a statistical model assuming an underlying factor structure generating the data. FA is based on a probabilistic model.
        *   **Pros:** Provides a probabilistic model, can be used to understand underlying constructs.
        *   **Cons:** Relies on model assumptions (linearity, factor structure), interpretations of factors can be subjective.

    *   **5.1.4 Independent Component Analysis (ICA)**
        *   **Type:** Unsupervised Linear Feature Extraction.
        *   **Goal:** To find a linear representation of non-Gaussian data where the components are statistically independent.
        *   **Core Idea:** Unlike PCA (which finds uncorrelated components), ICA seeks *independent* components. Independence is a stronger condition than uncorrelatedness (though for Gaussian variables, they are equivalent). Useful for separating mixed signals (e.g., Blind Source Separation, like separating voices in a crowded room).
        *   **Pros:** Useful for signal separation, finds components that are statistically independent.
        *   **Cons:** Assumes non-Gaussian sources, order and scale of components are not uniquely determined.

    *   **Non-linear Dimensionality Reduction (Manifold Learning):** These techniques aim to preserve the non-linear structure (manifold) of the data when projecting to a lower dimension.

        *   **5.1.5 Locally Linear Embedding (LLE)**
            *   **Type:** Unsupervised Non-linear Feature Extraction.
            *   **Goal:** To preserve the local structure of the data. It assumes that each data point can be linearly reconstructed from its neighbors.
            *   **Core Idea:** For each data point, find its neighbors and the weights that linearly reconstruct the point from its neighbors. Then, find a low-dimensional representation where each point is *still* reconstructed by the same weights from its neighbors.
            *   **Pros:** Good at preserving local neighborhood structure, can uncover complex non-linear structures.
            *   **Cons:** Sensitive to the number of neighbors (k), computationally expensive, struggle with highly varying densities, sensitive to noise.

        *   **5.1.6 Isomap**
            *   **Type:** Unsupervised Non-linear Feature Extraction.
            *   **Goal:** To preserve the geodesic distances (distances along the manifold) between data points.
            *   **Core Idea:** Uses the shortest path distances on a neighborhood graph of the data points (approximating the geodesic distance) and then applies an algorithm like Multidimensional Scaling (MDS) to find a low-dimensional embedding that preserves these graph distances.
            *   **Pros:** Aims to preserve global structure (geodesic distances), can uncover complex non-linear structures.
            *   **Cons:** Sensitive to the number of neighbors (k) used to build the graph, building the neighborhood graph can be computationally expensive.

    *   **5.1.7 Least Squares Optimization**
        *   **Context:** While not a dimensionality reduction *technique* itself, Least Squares Optimization is a fundamental principle *used by* many algorithms, including some dimensionality reduction methods or steps within them (e.g., the projection step in PCA can be derived from a least-squares perspective minimizing reconstruction error, or fitting steps in Factor Analysis models).
        *   **Definition:** A method for finding the best fit to data by minimizing the sum of the squares of the differences (errors) between the observed values and the values predicted by a model.
        *   **Goal:** To find model parameters that minimize the squared error.
        *   **Example:** In Linear Regression (Unit 1), the goal is to find the weights and bias that minimize the Mean Squared Error (MSE), which is the average of squared errors. This minimization can often be solved using least squares methods.

### 5.2 Evolutionary Learning

*   **Concept:** A set of optimization and learning algorithms inspired by the process of natural evolution (natural selection, mutation, reproduction).
*   **Core Idea:** Maintain a population of candidate solutions. In each generation, select the "fittest" solutions, apply genetic operators (mutation, crossover) to create new solutions (offspring), and replace some or all of the old population with the new one. Repeat over generations, hoping the population evolves towards better solutions.
*   **Good for:** Optimization problems, search problems, and sometimes for generating machine learning models or tuning their parameters.

*   **5.2.1 Genetic Algorithms (GAs)**
    *   **Type:** A popular type of Evolutionary Algorithm.
    *   **Goal:** To find optimal or near-optimal solutions to problems by mimicking natural evolution.
    *   **Key Components:**
        *   **Population:** A collection of individuals (candidate solutions).
        *   **Individual (Chromosome/Genotype):** Represents a potential solution to the problem. Often encoded as a string of bits or other values.
        *   **Fitness Function:** Evaluates the quality or performance of each individual (how "fit" it is). This is the objective function to be optimized.
        *   **Selection:** Choose individuals from the current population to create the next generation. Fitter individuals have a higher probability of being selected.
        *   **Genetic Operators:** Applied to selected individuals to generate new offspring.

    *   **5.2.2 Genetic Offspring:** The new individuals created by applying genetic operators to selected parents.

    *   **5.2.3 Genetic Operators:**
        *   **Crossover (Recombination):** Combines genetic material from two or more parent individuals to create one or more new offspring. E.g., single-point crossover swaps segments of the chromosome from two parents.
        *   **Mutation:** Randomly alters one or more genes (parts of the chromosome) of an individual. Introduces variation into the population, helping to explore the search space and avoid getting stuck in local optima.

    *   **5.2.4 Using Genetic Algorithms:**
        *   **Procedure:**
            1. Initialize a population of random individuals.
            2. Evaluate the fitness of each individual in the population.
            3. Select individuals based on their fitness.
            4. Apply crossover and mutation operators to create a new generation of offspring.
            5. Replace the old population with the new population (or integrate them based on some criteria).
            6. Repeat steps 2-5 for a fixed number of generations or until a satisfactory solution is found.
        *   **Applications in ML:** Optimizing hyperparameters of ML models, evolving neural network architectures (Neuroevolution), feature selection, training specific types of classifiers.

### 5.3 Reinforcement Learning (RL)

*   **Concept:** A paradigm of machine learning where an **agent** learns to make decisions by taking **actions** in an **environment** to maximize a cumulative **reward**.
*   **Core Idea:** Learning is based on trial and error. The agent receives feedback from the environment in the form of rewards or penalties, and it learns a **policy** (a strategy for choosing actions) that leads to the highest accumulated reward over time.
*   **Key Components:**
    *   **Agent:** The learner or decision-maker.
    *   **Environment:** Everything outside the agent; it's the world the agent interacts with.
    *   **State (s):** A description of the current situation of the environment.
    *   **Action (a):** A decision or move made by the agent that affects the environment's state.
    *   **Reward (r):** A scalar value given by the environment to the agent after an action, indicating how good or bad the action was at that moment. The goal is to maximize *cumulative* reward.
    *   **Policy (π):** The agent's strategy for choosing actions given a state, denoted as P(a|s) or a=π(s). This is what the agent learns.
    *   **Value Function (V(s) or Q(s, a)):** Predicts the expected cumulative future reward from a given state (V) or taking a given action in a given state (Q). The agent often learns value functions to help determine its policy.

*   **Learning Process (Simplified):**
    1.  Agent observes the current state `s`.
    2.  Agent chooses an action `a` based on its current policy.
    3.  Environment transitions to a new state `s'` and provides a reward `r`.
    4.  Agent uses the experience (`s`, `a`, `r`, `s'`) to update its policy or value functions.
    5.  Repeat.

*   **Exploration vs. Exploitation:** A key challenge is balancing exploration (trying new actions to discover potentially better rewards) and exploitation (choosing actions known to yield high rewards based on current knowledge).

*   **5.3.1 Overview:** RL is distinct from supervised learning (no labeled correct actions provided, only rewards) and unsupervised learning (goal is to maximize reward, not just find structure). It's suitable for tasks involving sequential decision-making.

*   **5.3.2 Getting Lost Example:**
    *   Imagine an agent (e.g., a robot) in a maze.
    *   **State:** The agent's current position in the maze.
    *   **Actions:** Move North, South, East, West.
    *   **Environment:** The maze layout.
    *   **Reward:** Maybe +1 for reaching the goal, -1 for hitting a wall, 0 for moving freely.
    *   **Goal:** The agent needs to learn a policy (e.g., if at location A, move North; if at location B, move East) that maximizes the total reward received over an episode (finding the goal). Initially, it might wander randomly ("getting lost"), but over time, by experiencing rewards and penalties, it learns an efficient path.

*   **5.3.3 Markov Decision Process (MDP)**
    *   **Definition:** A mathematical framework used to model sequential decision-making problems where the outcomes are partly random and partly under the control of a decision-maker. Many Reinforcement Learning problems are modeled as MDPs.
    *   **Key Property: Markov Property:** The future state depends *only* on the current state and the action taken, not on the sequence of events that preceded the current state. P(s' | s, a, s₀, a₀, ..., sᵢ, aᵢ) = P(s' | s, a). This "memoryless" property simplifies the problem.
    *   **Components:** MDPs are defined by:
        *   A set of states S.
        *   A set of actions A.
        *   A transition probability function P(s' | s, a), the probability of reaching state s' from state s by taking action a.
        *   A reward function R(s, a, s'), the reward received after taking action a in state s and transitioning to state s'.
        *   (Optionally) A discount factor γ, which determines the importance of future rewards.
    *   **Solving an MDP:** Finding an optimal policy (π*) that maximizes the expected cumulative discounted reward from any state. Techniques like Dynamic Programming, Monte Carlo methods, and Temporal Difference learning (which RL algorithms use) are employed.

### 5.4 Summary of Unit 5

Unit 5 covered three distinct, but important, areas:

*   **Dimensionality Reduction:** Techniques to reduce the number of features. We discussed **PCA** (unsupervised, maximize variance), **LDA** (supervised, maximize class separation), **Factor Analysis** (model-based, uncover latent factors), **ICA** (independent components), and non-linear methods like **LLE** and **Isomap** (preserve local/geodesic structure). **Least Squares Optimization** was mentioned as an underlying principle in many methods.
*   **Evolutionary Learning:** Learning inspired by natural evolution, particularly **Genetic Algorithms**. Key concepts include population, fitness function, selection, and **Genetic Operators** (Crossover, Mutation) used to create **Genetic Offspring** iteratively improving solutions.
*   **Reinforcement Learning (RL):** A learning paradigm where an **agent** learns to make sequential **actions** in an **environment** by maximizing cumulative **rewards**. Key components include states, actions, rewards, policy, and value functions. We saw a simple **Getting Lost Example** and introduced the **Markov Decision Process (MDP)** as a formal model for RL problems, highlighting the Markov property.

This unit concludes our overview of core machine learning techniques, touching upon methods for handling complex data (dimensionality reduction), alternative optimization/learning paradigms (evolutionary learning), and learning through interaction (reinforcement learning).

---
---
---

 
 #  **concise, exam-focused summary of the entire BCA120 Machine Learning Techniques syllabus, designed for quick revision. It highlights key definitions, purposes, and relationships between concepts across the units**.

---

**BCA120: Machine Learning Techniques - Exam Revision Notes**

This summary covers the core concepts and techniques from all 5 units, focusing on what's essential for understanding and recalling for an exam.

**Unit 1: Introduction & Foundational Concepts**

*   **Machine Learning:** Learning from data without explicit programming (vs. rule-based).
*   **Types of Learning:**
    *   **Supervised:** Learn from **labeled** data (Input X -> Output y). Tasks: **Classification** (predict category), **Regression** (predict continuous value).
    *   **Unsupervised:** Learn from **unlabeled** data (Input X only). Tasks: **Clustering** (grouping), **Dimensionality Reduction**, Association.
    *   **Ensemble:** Combine multiple models for better performance/robustness.
*   **Hypothesis Space:** Set of all possible models an algorithm can learn.
*   **Inductive Bias:** Algorithm's assumptions/preferences to choose one hypothesis over others; necessary for generalization from finite data.
*   **Evaluation:** How well a model performs, especially on **unseen data** (generalization).
    *   **Train-Test Split:** Basic method (e.g., 80% train, 20% test). Test set kept separate.
    *   **Cross-Validation (K-Fold):** More robust. Split data into K folds, train K times using K-1 folds for training and 1 for testing, average results. Reduces variance in evaluation estimate.
*   **Linear Regression:** Supervised (Regression). Models linear relationship (y = wx + b), minimizes **Mean Squared Error (MSE)**.
*   **Decision Trees:** Supervised (Class/Reg). Tree structure, internal nodes are tests, leaves are predictions. Built greedily. Prone to overfitting.
*   **Overfitting:** Model learns training data too well (including noise), performs poorly on test data. (High train accuracy, low test accuracy).
    *   **Cause:** Model too complex for data.
    *   **Mitigation:** More data, simpler model, regularization, pruning (for trees), cross-validation.
*   **Design a Learning System:** Steps involved (Problem -> Data -> Prep -> FE -> Model -> Train -> Eval -> Tune -> Deploy -> Monitor).
*   **Feature Engineering:** Transforming raw data into useful features.

**Unit 2: Unsupervised Learning, Instance-Based, Probability**

*   **Unsupervised Learning (Deep Dive):** Finding hidden structure in unlabeled data.
*   **Clustering:** Grouping similar data points.
    *   **K-Means:** Partitioning. Requires *k*. Iterative: Assign points to nearest **centroid**, update centroid to mean of points. Simple, sensitive to init/k.
    *   **Hierarchical Clustering:** Builds a **dendrogram** (tree hierarchy). Agglomerative (bottom-up merge) or Divisive (top-down split). No *k* needed initially.
    *   **Gaussian Mixture Model (GMM):** Model-based clustering. Assumes data is mix of Gaussian distributions. Uses **EM algorithm**. Provides probabilistic (soft) cluster assignments.
*   **Vector Quantization:** Data compression using clustering (often K-Means), representing data by cluster centroids.
*   **Self Organizing Feature Map (SOM):** Neural network for unsupervised learning. Maps high-D data to low-D grid, preserving topology.
*   **Instance-Based Learning:** Stores training data. Prediction computes similarity to stored instances. (Lazy learning).
    *   **K-Nearest Neighbors (KNN):** (Mainly Supervised Class/Reg). Predicts based on majority class/average of *k* nearest training points. Sensitive to distance metric and *k*.
*   **Probability and Bayes Learning:** Mathematical framework for uncertainty. Basis for probabilistic models (GMM). Bayes' Theorem: P(A|B) = [P(B|A)P(A)]/P(B).
*   **Data into Probabilities:** Methods to estimate probabilities from data.
*   **Basic Statistics:** Mean, variance, correlation, etc. Essential for data analysis and understanding algorithms.
*   **Feature Reduction:** Reducing number of features (Selection or Extraction).

**Unit 3: Discriminative Models, Kernels, Neural Networks**

*   **Logistic Regression:** Supervised (Classification). Uses **Sigmoid function** to map linear output to probability [0,1]. Learns a linear decision boundary. Trained using **Cross-Entropy Loss**.
*   **Support Vector Machine (SVM):** Supervised (Classification). Finds the **maximum-margin hyperplane** separating classes. Key points are **Support Vectors**. Hard (strictly separable) vs Soft margin (allows some errors).
*   **Kernel Function and Kernel SVM:**
    *   **Kernel Trick:** Avoids explicit mapping to high-dimensional space. Computes dot products *as if* in higher dimension using a **Kernel Function** K(xᵢ, xⱼ).
    *   **Kernel SVM:** SVM using kernels (e.g., **RBF, Polynomial**) to learn non-linear decision boundaries in original space.
*   **Neural Network:** Inspired by biological neurons.
    *   **Perceptron:** Single artificial neuron. Linear classifier.
    *   **Multilayer Network (MLP):** Input, one or more Hidden, Output layers. Uses non-linear **activation functions**. Can learn complex non-linear patterns.
    *   **Back Propagation:** Algorithm to train MLPs. Uses chain rule to calculate gradient of loss w.r.t weights, enabling optimization (e.g., Gradient Descent) to update weights iteratively.
    *   **Deep Neural Network (DNN):** MLP with many hidden layers. Learns hierarchical features. Revolutionized many fields due to data/compute/algorithmic advances.

**Unit 4: Learning with Trees and Ensemble Methods**

*   **Learning with Trees:** Focus on Decision Trees.
*   **Constructing Decision Trees:** Greedy, recursive splitting process.
*   **Splitting Criteria:** Measure how well a split separates data: **Gini Impurity, Entropy/Information Gain** (Classification); **Variance Reduction** (Regression).
*   **Classification and Regression Trees (CART):** Specific algorithm using Gini (Class) and Variance (Reg), produces binary trees.
*   **Ensemble Learning:** Combine multiple base models. Improves performance, reduces variance/bias.
*   **Bagging (Bootstrap Aggregating):** Train base models on different **bootstrap samples** (sampling with replacement). Combine via **voting** (Class) or **averaging** (Reg). E.g., **Random Forest** (Bagging using Decision Trees + random feature subsetting). Primarily reduces **variance**.
*   **Boosting:** Train base models sequentially. Each focuses on errors of previous ones (by weighting data or targeting residual errors). Combine via weighted summing. E.g., **AdaBoost, Gradient Boosting**. Primarily reduces **bias**.
*   **Different ways to Combine Classifiers:** Voting (simple/weighted), Averaging, Stacking (meta-learner uses base predictions as features).
*   **Probability and Learning, Data into Probabilities, Basic Statistics, GMM, Nearest Neighbour Methods:** Reiteration of concepts from Unit 2, highlighting their fundamental role and application in various ML methods, including probabilistic models (GMM) and non-parametric methods (KNN).

**Unit 5: Dimensionality Reduction, Evolutionary Models, Reinforcement Learning**

*   **Dimensionality Reduction:** Reduce number of features while retaining information.
    *   **Why:** Curse of dimensionality, efficiency, visualization, noise reduction.
    *   **Feature Selection:** Choose subset of original features.
    *   **Feature Extraction:** Create new features (components).
    *   **Principal Component Analysis (PCA):** Unsupervised, linear. Finds orthogonal components maximizing variance.
    *   **Linear Discriminant Analysis (LDA):** Supervised, linear. Finds components maximizing class separation.
    *   **Factor Analysis (FA):** Unsupervised, model-based. Explains variance through fewer latent factors.
    *   **Independent Component Analysis (ICA):** Unsupervised. Finds statistically independent components (non-Gaussian data).
    *   **Locally Linear Embedding (LLE):** Unsupervised, non-linear. Preserves local neighborhood structure.
    *   **Isomap:** Unsupervised, non-linear. Preserves geodesic distances.
    *   **Least Squares Optimization:** Principle of minimizing sum of squared errors, used in many algorithms (Linear Regression, PCA, etc.).
*   **Evolutionary Learning:** Optimization/learning inspired by natural evolution.
    *   **Genetic Algorithms (GAs):** Population of solutions (**chromosomes**). Uses **fitness function** for selection. New solutions (**offspring**) created via **Genetic Operators** (**Crossover**, **Mutation**). Iterative improvement.
*   **Reinforcement Learning (RL):** Agent learns to make sequential decisions in an **environment** to maximize cumulative **reward**. Learning by trial and error.
    *   **Key elements:** **Agent**, **Environment**, **State**, **Action**, **Reward**, **Policy**.
    *   **Getting Lost Example:** Agent learns path in maze using rewards/penalties.
    *   **Markov Decision Process (MDP):** Formal framework for RL. Assumes **Markov Property** (future depends only on current state and action). Defined by states, actions, transition probabilities, rewards.

---

This summary should serve as a good checklist and quick reference for your exam preparation. Focus on understanding the core idea, input/output, and purpose of each algorithm and concept. Good luck!
