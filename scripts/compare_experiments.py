import pandas as pd
import json

# Constants for the file names

exp1 = '20231101194906'
exp2 = '20231102072822'
FILE1_PATH = f'experiments/{exp1}/results.csv'
FILE2_PATH = f'experiments/{exp2}/results.csv'
QUESTIONS_PATH = 'data/hotpot_dev_pretty.json'

# Adjust display options to avoid truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)  # Adjust the width to accommodate the number of columns

# Load the questions from the JSON file
with open(QUESTIONS_PATH, 'r') as file:
    questions_data = json.load(file)
# Convert the questions to a DataFrame
questions_df = pd.DataFrame(questions_data)

# Read the two CSV files into pandas DataFrames
df1 = pd.read_csv(FILE1_PATH)
df1['question_text'] = df1['question_index'].apply(lambda idx: questions_df.iloc[idx]['question'])
df2 = pd.read_csv(FILE2_PATH)
df2['question_text'] = df2['question_index'].apply(lambda idx: questions_df.iloc[idx]['question'])


# Columns to compare between the files
merge_columns = ['chunk_size', 'prompt', 'example_chunk_size', 'max_llm_calls', 'model', 'question_text']

# Merge the two DataFrames on the key columns
merged_df = pd.merge(df1, df2, on=merge_columns, suffixes=('_file1', '_file2'))
# Map the question text to each row in the merged DataFrame
# Find rows where the 'correct' column does not match
diff_df = merged_df[merged_df['correct_file1'] != merged_df['correct_file2']]

# Output the rows with different 'correct' values
if not diff_df.empty:
    print("Rows with differing 'correct' values:")
    print(diff_df[merge_columns + ['correct_file1', 'correct_file2']])
else:
    print("No rows with differing 'correct' values.")

# Find common rows based on the specified columns
common_rows = pd.merge(df1[merge_columns], df2[merge_columns], how='inner')

# Filter the original dataframes to only include the common rows
filtered_df1 = pd.merge(common_rows, df1, on=merge_columns, how='left')
filtered_df2 = pd.merge(common_rows, df2, on=merge_columns, how='left')

# Sum the 'correct' column in both filtered dataframes
sum_correct_df1 = filtered_df1['correct'].sum()
sum_correct_df2 = filtered_df2['correct'].sum()

# Output the sums to the console
print(f"Sum of 'correct' in {FILE1_PATH} for common rows: {sum_correct_df1}")
print(f"Sum of 'correct' in {FILE2_PATH} for common rows: {sum_correct_df2}")

# Compare the sums and print the result
if sum_correct_df1 > sum_correct_df2:
    print(f"{FILE1_PATH} has a greater sum of 'correct'.")
elif sum_correct_df1 < sum_correct_df2:
    print(f"{FILE2_PATH} has a greater sum of 'correct'.")
else:
    print("Both files have the same sum of 'correct' for the common rows.")
