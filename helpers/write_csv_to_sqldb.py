import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect('./data/db/employee_leaves.db')

# Load CSV data into a pandas DataFrame
df = pd.read_csv('./data/raw_data/db_data/emp_leave_info.csv')

# Write the data to a SQLite table
df.to_sql('employee_leaves_data', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

#===================================================================================================
## Read and find datatype

# # Connect to the SQLite database
# conn = sqlite3.connect('./data/db/leaves.db')
# c = conn.cursor()
#
# # Get the table structure
# c.execute("PRAGMA table_info(leave_data)")
# columns = c.fetchall()
#
# # Print column names and types
# for column in columns:
#     print(f"Column: {column[1]}, Type: {column[2]}")

# # Recast column to date
# c.execute("""
#     CREATE TABLE new_table AS
#     SELECT
#         DATE(Date_and_Time_of_Event) as Date_and_Time_of_Event,
#         Reporter_Name,
#         Reporter_Title,
#         Reporter_ Contact_Information,
#         Patient_Name,
#         Age,
#         Sex,
#         Weight_Kg,
#         Medical_History,
#         Month_of_Event,
#         Year_of_Event,
#         Adverse_Event_Description,
#         Product_Brand_Name,
#         Product_Generic_Name,
#         Batch_Lot_Number,
#         Expiration_Date,
#         Dosage,
#         Severity,
#         Outcome,
#         Concomitant_Medications,
#         Causal_Relationship_Assessment,
#         Follow-up Information,
#         Report_Submission_Deadline_Met
#
#     FROM adverse_events
# """)
#
# # Delete the old table
# c.execute("DROP TABLE adverse_events")
#
# # Rename the new table to the old table name
# c.execute("ALTER TABLE new_table RENAME TO adverse_events")
#
# # Commit the changes and close the connection
# conn.commit()
# conn.close()
