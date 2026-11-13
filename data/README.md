# Data

| File | Used in | Included in this repo? | Source / licence |
|---|---|---|---|
| `Student_List_A2.csv` | Part A (A1–A5): labelled training data, 2,100 students | **No** | FIT1043 course material (Monash Moodle) |
| `Student_List_A2_Submission.csv` | Part A (A6): unlabelled Kaggle competition data, 161 students | **No** | FIT1043 course material (Monash Moodle) |
| `penguins.csv` | Part B: K-means clustering, 344 penguins | Yes | [Palmer Penguins Dataset for EDA](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda) (Kaggle, MIT); original data from [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) (CC0) |

## Student performance data (not included)

The two student files were provided to enrolled students through the FIT1043 Moodle site. They are a
**modified** version of Rabie El Kharoua's *Students Performance Dataset*
([Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset),
DOI [10.34740/kaggle/ds/5195702](https://doi.org/10.34740/kaggle/ds/5195702), CC BY 4.0). The course
version was reduced to a subset of columns and seeded with data-quality problems (missing study times,
impossible absence counts, and grade labels that contradict GPA) for the wrangling exercises. It also holds
the unlabelled competition data, so it is not redistributed here.

Schema expected by the notebook:

| Column | Type | Description |
|---|---|---|
| `StudentID` | int | Identifier (1002–3392 training, 5000–5160 submission) |
| `Age` | int | 15–18 |
| `StudyTimeWeekly` | float | Weekly study time in hours (21 missing values in the training file) |
| `Absences` | int | Number of absences (two invalid values: −122 and 320) |
| `ParentalSupport` | int | 0 (none) to 4 (very high) |
| `GPA` | float | 0.0–4.0, training file only |
| `GradeClass` | int | **Target.** 0 = A, 1 = B, 2 = C, 3 = D, 4 = F, training file only |

To run Part A, place both course files in this folder (they are git-ignored). Part B (clustering) runs
without them.

The public Kaggle dataset is the upstream source, **not** a drop-in replacement: it has additional columns
(which the notebook's `drop(...)` would pass through as features), none of the injected data-quality issues,
and no competition rows. Results obtained from it would not be comparable to those reported here.

## Palmer Penguins (included)

`penguins.csv` is the 344-row Palmer Penguins table with an added `id` column, as downloaded from the
Kaggle dataset linked above. Data were collected by Dr. Kristen Gorman and the Palmer Station LTER.

> Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data.*
> R package version 0.1.0. https://allisonhorst.github.io/palmerpenguins/. doi:10.5281/zenodo.3960218
