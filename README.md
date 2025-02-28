# Online Course Engagement Analysis

## Project Overview
This project focuses on analyzing student engagement in an online learning platform. A machine learning model is used to predict course completion rates based on user behavior, quiz performance, and video interactions. The goal is to identify key engagement factors and enhance learning outcomes.

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
python src/preprocessing.py
```

### 3. Feature Selection
Execute the feature selection script to identify the most relevant features:
```bash
python src/feature_selection.py
```

### 4. Model Training
Train the model using the following command:
```bash
python src/modelling.py
```

### 5. Model Inference
To make predictions using the trained model, run:
```bash
python src/inference.py
```

---

## Model Performance Summary
### Key Insights from Feature Importance Analysis:
The model identified the following as the most important features influencing student engagement and course completion:

1. **Completion Rate** - The strongest predictor of success; students with higher engagement are more likely to complete the course.
2. **Quiz Scores** - Higher quiz performance correlates with better retention and engagement.
3. **Number of Videos Watched** - Indicates active participation in the learning process.
4. **Number of Quizzes Taken** - Reflects a student’s willingness to test their understanding.
5. **Time Spent on Course** - Students who dedicate more time have better outcomes.
6. **Engagement Rate** - A metric combining multiple engagement indicators.
7. **Quiz Score Per Attempt** - Measures improvement and effort in assessments.
8. **Course Category** - Certain courses may have varying levels of engagement.
9. **Low Effort User** - Identifies users with minimal interaction who are at risk of disengagement.

### Model Performance Metrics:
- **Precision:** 91% (Only 9% of flagged cases are misclassified)
- **Recall:** 79% (Most engaged users are correctly classified)
- **Accuracy:** 87% (Strong overall model performance)

These insights help improve course design by focusing on content and engagement strategies that maximize completion rates.

---

## Author
- **Olajide** - Data Scientist & Machine Learning Engineer

For questions, contact yusufolajideda1@gmail.com.

