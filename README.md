## Online Course Completion Prediction

## Project Overview
This project aims to predict whether a user will complete an online course based on engagement metrics. Using machine learning techniques, the model identifies key factors influencing course completion, allowing for better interventions to improve student retention and engagement.

---

## Project Structure
```
DEXTER_CYBERLAB/
│-- data/                         # Dataset storage
│   ├── online_course_engagement.csv   # Raw dataset
│-- src/                          # Python scripts for different stages
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   ├── feature_selection.py       # Recursive Feature Elimination (RFE)
│   ├── modelling.py               # Model training script
│   ├── inference.py               # Script for making predictions
│-- main.ipynb                     # Exploratory data analysis and visualization
│-- requirements.txt                # Dependencies list
│-- .gitignore                      # Files to be ignored in version control
│-- README.md                       # Project documentation
```

---



## How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed. Install required libraries using:
```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing
Run the preprocessing script to clean and transform the dataset:
```bash
python scripts/data_preprocessing.py
```

### 3. Model Training
Train the model using the following command:
```bash
python scripts/model_training.py
```

### 4. Model Inference
To make predictions using the trained model, run:
```bash
python scripts/inference.py
```

---

## Model Performance Summary
### RandomForest Model Performance
- **Threshold:** The best threshold for classification was set to **0.35**.
- **Precision:** 95% (Only 5% of predicted completions were incorrect, reducing false positives).
- **Recall:** 92% (Most course completions were detected, ensuring good engagement tracking).
- **Accuracy:** 95% (Overall model performance is strong in predicting course completion correctly).

### Key Insights from Exploratory Data Analysis:
- **Completion Rate:** The most influential factor. Users with higher completion rates tend to finish courses successfully.
- **Quiz Scores:** Higher quiz scores correlate with an increased likelihood of completing a course.
- **Number of Videos Watched & Quizzes Taken:** These engagement metrics show a strong positive impact on completion likelihood.
- **Time Spent on Course & Engagement Rate:** More time invested leads to higher completion probabilities.
- **Quiz Score Per Attempt:** A moderate indicator, suggesting that students who perform well on quizzes are more likely to persist.
- **Course Category & Low Effort Users:** Have minimal impact on predicting completion.

The insights suggest that user engagement—especially completion rate, quiz scores, and time spent—is crucial for predicting course completion. Course providers can use this information to target interventions for students at risk of dropping out.

---

## Author
- **Olajide** - Data Scientist & Machine Learning Engineer

For questions, contact yusufolajideda1@gmail.com.

