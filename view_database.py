import sqlite3

conn = sqlite3.connect("student_performance.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:")
print(cursor.fetchall())

table = input("Enter table name: ")

cursor.execute(f"SELECT * FROM {table}")
rows = cursor.fetchall()

print("\nData:")
for row in rows:
    print(row)

conn.close()