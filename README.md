# Student Performance Analysis & Machine Learning

Predicting students' grade classes from attendance, study habits and background, with K-means clustering
on the Palmer Penguins dataset as a second experiment.

The project works through a full data science workflow in a single Jupyter notebook. It starts with a
student-records dataset that has missing values, impossible values and labels that contradict each other.
The data is cleaned and validated, explored for the factors that drive academic performance, and used to
train and compare three classifiers. The best model's predictions were submitted to a Kaggle competition.
Separately, an unsupervised experiment tests whether K-means can recover penguin species from body
measurements alone.

The work began as a project for FIT1043 Introduction to Data Science at Monash University Malaysia
(Semester 2, 2025).

## Contents

1. [Key results](#key-results)
2. [Technologies](#technologies)
3. [Problem and data](#problem-and-data)
4. [Data cleaning and validation](#data-cleaning-and-validation)
5. [Exploratory data analysis](#exploratory-data-analysis)
6. [Models and evaluation](#models-and-evaluation)
7. [Kaggle competition results](#kaggle-competition-results)
8. [Unsupervised learning: Palmer Penguins](#unsupervised-learning-palmer-penguins)
9. [Key findings and limitations](#key-findings-and-limitations)
10. [Installation and reproducibility](#installation-and-reproducibility)
11. [Project structure](#project-structure)

## Key Results

- **Kaggle: 9th on the MUM FIT1043 S2 2025 leaderboard**, with a score of **0.7701** across 23 submissions.
  The submitted model was an RBF-kernel SVM.
- **Local hold-out accuracy of about 74%** for all three classifiers on 315 unseen students: Gradient Boosting
  73.97%, SVM 73.65%, Random Forest 73.33%.
- **Absences are the dominant signal.** Their correlation with GPA is r = −0.92, compared with r = 0.18 for
  weekly study time.
- **K-means recovers penguin species** from two measurements, with an Adjusted Rand Index of **0.874**.

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter Notebook

## Problem and Data

**Task:** multi-class classification of `GradeClass`, a student's letter grade (A, B, C, D or F, encoded 0–4).

| Dataset | Rows | Role |
|---|---|---|
| `Student_List_A2.csv` | 2,100 | Labelled training data |
| `Student_List_A2_Submission.csv` | 161 | Unlabelled competition data, used for the Kaggle predictions |
| `penguins.csv` | 344 | Palmer Penguins, used for clustering |

The student data is a modified version of Rabie El Kharoua's
[Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset),
distributed to the course. It is not included in this repository. See [`data/README.md`](data/README.md)
for its source, schema and availability.

**Features:** `Age`, `StudyTimeWeekly`, `Absences`, `ParentalSupport`.

Two columns are excluded on purpose:

- **`StudentID`** is an identifier and carries no predictive information.
- **`GPA`** is excluded to avoid target leakage. Each grade class is defined directly by a GPA band, so a
  model given GPA would simply learn the grading thresholds.

## Data Cleaning and Validation

The raw training data contains three kinds of data-quality problem, each detected and handled explicitly.

| Issue | Detection | Treatment |
|---|---|---|
| 21 missing `StudyTimeWeekly` values | `isnull().sum()` | Median imputation, which is robust to outliers |
| Impossible `Absences` values (−122 and 320) | `describe()` and a boxplot | Dropped the 2 rows outside the valid 0–30 range, leaving 2,098 rows |
| 100 rows whose `GradeClass` contradicts their GPA | Recomputed the expected grade from GPA bands (A ≥ 3.5, B ≥ 3.0, C ≥ 2.5, D ≥ 2.0, F < 2.0) | Treated GPA as the source of truth and relabelled; a re-audit confirms 0 remaining inconsistencies |

The same imputation and validity checks are applied to the competition data before prediction. It
contained no missing or invalid values, so all 161 students received predictions.

<img src="images/absences-boxplot.png" alt="Boxplot of Absences showing outliers at -122 and 320" width="480">

## Exploratory Data Analysis

### Class distribution

The target is heavily imbalanced: 53.6% of students are F, 17.6% D, 16.2% C, 9.7% B and only 2.9% A.
This shapes how the models should be evaluated, because a classifier can score well on accuracy while
performing poorly on the rare classes.

<img src="images/grade-distribution-pie.png" alt="Pie chart of GradeClass proportions" width="380">

### Relationships with GPA

| Feature | Pearson r with GPA | Interpretation |
|---|---|---|
| `Absences` | **−0.920** | Very strong negative relationship |
| `StudyTimeWeekly` | 0.179 | Weak positive relationship |

<p>
<img src="images/absences-vs-gpa.png" alt="Scatter plot of absences against GPA" width="420">
<img src="images/study-time-vs-gpa.png" alt="Scatter plot of weekly study time against GPA" width="420">
</p>

Grouping by grade class makes the pattern concrete. Median absences rise steadily: 1 (A), 3 (B), 6 (C),
11 (D) and 21 (F). The relationship is strong, but it is not evidence of causation. Health, motivation or
prior knowledge could drive both absences and grades, and the notebook discusses what extra data would be
needed to separate these effects.

## Models and Evaluation

### Preprocessing

- Stratified 85/15 train/test split (`random_state=42`): 1,783 training rows and 315 test rows. Stratifying
  preserves the class balance in both sets.
- `StandardScaler` is fitted on the training set only and then applied to the test and competition data,
  which prevents information leaking from the evaluation data.
- Scaling matters most for the SVM, which is distance-based. Unscaled, `Absences` (0–29) would dominate
  `Age` (15–18).

### Models

| Model | Configuration |
|---|---|
| Support Vector Machine | `SVC(kernel='rbf', C=1.0, gamma='scale')` |
| Random Forest | 100 trees, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `max_features='sqrt'`, `class_weight='balanced'`, entropy criterion |
| Gradient Boosting | 300 stages, `learning_rate=0.1`, `max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10` |

### Hold-out test results (315 students)

| Model | Train accuracy | Test accuracy | Correct | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 100.0% | **73.97%** | 233 / 315 | 0.58 | 0.56 | 0.57 | 0.74 |
| SVM | 77.73% | **73.65%** | 232 / 315 | 0.50 | 0.51 | 0.50 | 0.72 |
| Random Forest | 77.96% | **73.33%** | 231 / 315 | 0.58 | 0.61 | 0.59 | 0.74 |

Per-class F1 scores (test support in brackets):

| Model | A (9) | B (31) | C (51) | D (55) | F (169) |
|---|---|---|---|---|---|
| Gradient Boosting | 0.38 | 0.42 | 0.55 | 0.56 | 0.93 |
| SVM | 0.00 | 0.60 | 0.59 | 0.40 | 0.92 |
| Random Forest | 0.45 | 0.46 | 0.53 | 0.58 | 0.93 |

### Interpretation

- **The accuracy gap between models is negligible.** Each model is one correctly classified student away
  from the next, so this test set cannot reliably rank them.
- **Gradient Boosting overfits.** It fits the training set perfectly and loses 26 percentage points on unseen
  data. SVM and Random Forest show train/test gaps of only about 4–5 points.
- **Most errors fall on the D/F boundary.** Students just above and below a GPA of 2.0 have very similar
  absence and study-time profiles, and the confusion matrices show all three models mixing these classes.
- **The rare A class is the hardest.** SVM never predicts an A (F1 = 0.00). Random Forest, which uses
  `class_weight='balanced'`, handles it best (F1 = 0.45) and has the highest macro F1 overall.

## Kaggle Competition Results

![Kaggle leaderboard: rank 9, Surya Muni Peragas, score 0.7701, 23 submissions](images/kaggle-leaderboard.png)

| | |
|---|---|
| Competition | MUM FIT1043 S2 2025 (In-Class Competition) |
| Leaderboard position | **9th** |
| Score | **0.7701** |
| Submissions | 23 |
| Final model | SVM (RBF kernel) |

The model ranked 9th on the leaderboard with a score of 0.7701. Several entries share identical scores, so
this is the leaderboard placement rather than a uniquely ninth-best score.

**The competition results differ from the local ones.** Gradient Boosting had the highest local test accuracy
(73.97% against 73.65% for SVM), but SVM performed best on Kaggle's held-out evaluation data and was used
for the final submission. That outcome fits the local diagnostics:

- The local ranking rests on single-sample differences, so it is effectively a tie.
- SVM's small train/test gap suggested it would generalise to new students. Gradient Boosting's 100%
  training accuracy signalled overfitting that would not carry over.

The leaderboard figures are historical results from the competition. The evaluation labels are private, so
the Kaggle score cannot be recomputed locally. The submitted predictions are in
[`outputs/kaggle_submission.csv`](outputs/kaggle_submission.csv), and running the notebook regenerates the
file exactly.

## Unsupervised Learning: Palmer Penguins

This experiment asks whether K-means, given no species labels, can rediscover the three penguin species from
body measurements.

**Data:** [Palmer Penguins](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda),
344 penguins of three species (Adelie, Chinstrap and Gentoo).

- **Missing data:** each of the four body measurements has 2 missing values, filled with the median. `sex`
  has 11 missing values, filled with the mode.
- **Features:** `bill_length_mm` and `flipper_length_mm`, standardised. Without scaling, flipper length
  (about 172–231 mm) would dominate bill length (about 32–60 mm) in the Euclidean distance.
- **k = 3**, matching the number of species. The species column is held back and used only to evaluate the
  clusters.

<p>
<img src="images/penguins-raw.png" alt="Raw scatter of bill length vs flipper length" width="420">
<img src="images/penguins-kmeans-clusters.png" alt="K-means clusters with centroids marked as stars" width="420">
</p>

**Results:** the expectation was that Gentoo penguins, the largest species, would form a clearly separate
cluster, with some overlap between Adelie and Chinstrap. The clustering matches this. The teal cluster
captures the long-flippered Gentoo penguins, and the remaining two clusters separate Adelie from Chinstrap
with a small overlap. A cross-tabulation against the true species confirms the picture:

| Cluster | Adelie | Chinstrap | Gentoo |
|---|---|---|---|
| 0 (purple) | **146** | 5 | 0 |
| 1 (teal) | 1 | 4 | **122** |
| 2 (yellow) | 5 | **59** | 2 |

The Adjusted Rand Index against the species labels is **0.874**. Using only two measurements, 327 of the 344
penguins fall in the cluster dominated by their own species.

## Key Findings and Limitations

**Findings**

- Attendance is by far the strongest predictor of performance in this data. Study time contributes much less.
- The three classifiers reach almost the same accuracy (about 74%). They differ in how they generalise and
  in how they treat the rare classes.
- Lower training error did not mean better real-world performance. The overfitted Gradient Boosting model
  led locally, but the better-regularised SVM did best on the competition data.

**Limitations**

- **Small evaluation sets.** With 315 test students, one prediction moves accuracy by about 0.3 points.
  K-fold cross-validation would give a more reliable model comparison than a single split.
- **Different evaluation data.** The local accuracy (about 74%) and the Kaggle score (0.7701) are computed on
  different students and are not directly comparable.
- **Class imbalance.** Accuracy is dominated by the F class (54% of students). The macro and per-class
  metrics show that SVM's strong competition result coexists with an inability to predict A grades.
- **Manual hyperparameters.** Model settings were chosen by hand rather than through a systematic search such
  as `GridSearchCV`.
- **Row filtering on new data.** The validity check on the competition data drops rows with invalid absences.
  No rows were affected here, but a production pipeline should clip or flag such values instead, so that
  every student still receives a prediction.
- **Data availability.** The student datasets are course-provided and not redistributed, so the
  classification analysis needs those files. The clustering analysis is fully reproducible from this
  repository.

## Installation and Reproducibility

Requires Python 3.11 or newer. The analysis was originally run on Python 3.13 and re-verified on Python 3.14
with the versions listed in `requirements.txt`.

```bash
git clone https://github.com/NotSury/student-performance-ml-analysis.git
cd student-performance-ml-analysis
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook student_performance_analysis.ipynb
```

Launch Jupyter from the repository root, because the notebook reads from `data/` and writes to `outputs/`
using relative paths.

| What you want to run | What you need |
|---|---|
| Browse the full analysis | Nothing. The notebook's saved outputs and figures come from a complete run. |
| Clustering (sections B1 onwards) | Nothing extra. `data/penguins.csv` is included. Select the first cell under **B1** and use *Run → Run Selected Cell and All Below*. |
| Classification and Kaggle predictions (sections A1–A6) | Place `Student_List_A2.csv` and `Student_List_A2_Submission.csv` in `data/` (see [`data/README.md`](data/README.md)), then *Run All*. This regenerates `outputs/kaggle_submission.csv`. |

## Project Structure

```
student-performance-ml-analysis/
├── student_performance_analysis.ipynb   # full analysis: classification (A1–A6) and clustering (B1)
├── data/
│   ├── README.md                        # dataset sources, licences and schema
│   └── penguins.csv                     # Palmer Penguins (included)
├── images/                              # figures exported from the notebook, plus the Kaggle leaderboard
├── outputs/
│   └── kaggle_submission.csv            # submitted SVM predictions (161 students)
└── requirements.txt
```
