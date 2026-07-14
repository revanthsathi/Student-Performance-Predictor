import numpy as np
import pandas as pd

np.random.seed(42)

number_of_students = 1000

study_hours = np.random.uniform(1, 10, number_of_students)
attendance = np.random.uniform(50, 100, number_of_students)
previous_marks = np.random.uniform(35, 100, number_of_students)
sleep_hours = np.random.uniform(4, 10, number_of_students)
assignments_completed = np.random.randint(0, 11, number_of_students)
class_participation = np.random.randint(1, 11, number_of_students)
screen_time = np.random.uniform(1, 10, number_of_students)
extracurricular_hours = np.random.uniform(0, 5, number_of_students)
stress_level = np.random.randint(1, 11, number_of_students)

final_score = (
    study_hours * 2.5
    + attendance * 0.20
    + previous_marks * 0.45
    + sleep_hours * 1.2
    + assignments_completed * 1.0
    + class_participation * 0.5
    - screen_time * 0.8
    + extracurricular_hours * 0.3
    - stress_level * 0.5
)

noise = np.random.normal(0, 5, number_of_students)

final_score = final_score + noise

final_score = np.clip(final_score, 0, 100)

data = pd.DataFrame({
    "study_hours": study_hours.round(2),
    "attendance": attendance.round(2),
    "previous_marks": previous_marks.round(2),
    "sleep_hours": sleep_hours.round(2),
    "assignments_completed": assignments_completed,
    "class_participation": class_participation,
    "screen_time": screen_time.round(2),
    "extracurricular_hours": extracurricular_hours.round(2),
    "stress_level": stress_level,
    "final_score": final_score.round(2)
})

data.to_csv("student_data.csv", index=False)

print("Student dataset generated successfully!")
print("Total students:", len(data))

print("\nSample Data:")
print(data.head())