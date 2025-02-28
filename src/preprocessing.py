import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/M.I.S SITHLTD/Desktop/olajide/Dexter_Cyberlab/data/online_course_engagement_data.csv")

# Display basic info
print(df.head(), "\n", df.info(), "\nDataset Shape:", df.shape)

# Visualizations
df['CourseCompletion'].value_counts(normalize=True).plot(kind="bar", title="Course Completion Distribution")
plt.show()

sns.countplot(x="CourseCategory", hue="CourseCompletion", data=df)
plt.title("Course Completion by Category")
plt.xticks(rotation=45)
plt.show()

# Feature Engineering
df["EngagementRate"] = df["TimeSpentOnCourse"] / (df["CompletionRate"] + 1e-6)
df["QuizScorePerAttempt"] = df["QuizScores"] / (df["NumberOfQuizzesTaken"] + 1e-6)
df["LowEffortUser"] = ((df["TimeSpentOnCourse"] < 10) & (df["CompletionRate"] < 20)).astype(int)

# Drop unnecessary columns
drop_cols = ["UserID", "DeviceType"]
df = df.drop(columns=drop_cols)

# Save cleaned data
df.to_csv("C:/Users/M.I.S SITHLTD/Desktop/olajide/Dexter_Cyberlab/data/preprocessed_data.csv", index=False)