---
name: ml-pipeline
description: "Build ML pipelines: data preparation, feature engineering, model training, evaluation, and deployment scaffolding."
source: community
allowed-tools: "*"
user-invocable: true
---

# ML Pipeline Builder

Scaffold a machine learning pipeline from data preparation through model deployment.

## STEP 1: DEFINE THE ML TASK

Parse $ARGUMENTS for:
- **Problem type**: Classification, regression, clustering, NLP, computer vision, recommendation
- **Data source**: Where training data comes from
- **Target variable**: What to predict or optimize
- **Success metric**: Accuracy, F1, RMSE, AUC, business metric
- **Deployment target**: Batch prediction, real-time API, edge device

## STEP 2: DATA PREPARATION

Design the data pipeline:

### Data Collection
- Source connections and extraction
- Data format and schema
- Sampling strategy if data is large

### Data Cleaning
- Handle missing values (impute, drop, flag)
- Remove duplicates
- Fix data types and encodings
- Handle outliers

### Exploratory Data Analysis
- Distribution of target variable
- Feature correlations
- Class imbalance assessment
- Data quality summary

## STEP 3: FEATURE ENGINEERING

Design features:

- **Numerical**: Scaling, normalization, binning, polynomial features
- **Categorical**: One-hot encoding, label encoding, target encoding
- **Text**: Tokenization, embeddings, TF-IDF
- **Temporal**: Time-based features, lag features, rolling statistics
- **Domain-specific**: Custom features based on domain knowledge

Create a feature pipeline that's reproducible and versioned.

## STEP 4: MODEL TRAINING

Set up the training pipeline:

- Train/validation/test split strategy (random, temporal, stratified)
- Model selection (baseline models, candidate models)
- Hyperparameter tuning strategy (grid search, random search, Bayesian)
- Cross-validation configuration
- Training infrastructure (local, cloud, GPU requirements)

## STEP 5: EVALUATION

Design evaluation:

- Metrics computation on test set
- Confusion matrix / error analysis
- Feature importance analysis
- Model comparison (baseline vs. candidates)
- Bias and fairness assessment if applicable
- Performance on edge cases and subgroups

## STEP 6: DEPLOYMENT SCAFFOLD

Set up deployment:

- Model serialization and versioning
- Prediction API or batch job scaffold
- Input validation and preprocessing
- Output post-processing
- Monitoring (data drift, prediction drift, performance degradation)
- A/B testing framework for model comparison
- Rollback strategy

## STEP 7: OUTPUT

Provide:
- Complete pipeline code with clear module boundaries
- Configuration files for reproducibility
- Documentation of design decisions
- Evaluation results and recommendations
