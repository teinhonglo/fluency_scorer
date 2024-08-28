import os
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process PCC data from a CSV file.")
parser.add_argument("file_path", type=str, help="Path to the CSV file")
args = parser.parse_args()

# Load the CSV file specified by the argument
data = pd.read_csv(args.file_path, sep=",")

# Dynamically identify train and test PCC columns
#train_pcc_columns = [col for col in data.columns if 'train_pcc' in col]
#test_pcc_columns = [col for col in data.columns if 'test_pcc' in col]
train_pcc_columns = sorted([col for col in data.columns if 'train_pcc' in col], key=lambda x: ('phone' in x, 'word' in x, 'utt' in x), reverse=True)
test_pcc_columns = sorted([col for col in data.columns if 'test_pcc' in col], key=lambda x: ('phone' in x, 'word' in x, 'utt' in x), reverse=True)

if "phone_train_mse" in data.columns:
    train_pcc_columns = ["phone_train_mse"] + train_pcc_columns
if "phone_test_mse" in data.columns:
    test_pcc_columns = ["phone_test_mse"] + test_pcc_columns

# Create separate dataframes for train and test
train_pcc_summary = data[train_pcc_columns]
test_pcc_summary = data[test_pcc_columns]

# Rename the index to reflect the first four rows (mean, std, max, min)
train_pcc_summary.index = ['mean', 'std', 'max', 'min']
test_pcc_summary.index = ['mean', 'std', 'max', 'min']

# Flatten any potential MultiIndex columns
if isinstance(train_pcc_summary.columns, pd.MultiIndex):
    train_pcc_summary.columns = ['_'.join(col).strip() for col in train_pcc_summary.columns.values]
if isinstance(test_pcc_summary.columns, pd.MultiIndex):
    test_pcc_summary.columns = ['_'.join(col).strip() for col in test_pcc_summary.columns.values]

# Function to convert a dataframe to markdown
def dataframe_to_markdown(df):
    return df.to_markdown()

# Convert the train and test PCC tables to markdown format
train_pcc_markdown = dataframe_to_markdown(train_pcc_summary)
test_pcc_markdown = dataframe_to_markdown(test_pcc_summary)

# Extract directory and base file name
dir_name = os.path.dirname(args.file_path)
base_file_name = os.path.splitext(os.path.basename(args.file_path))[0]

# Generate file paths for saving markdown files in the same directory
markdown_file_path = os.path.join(dir_name, f"results.md")

if os.path.isfile(markdown_file_path):
    mode = "a"
else:
    mode = "w"
# Save the markdown tables to files
with open(markdown_file_path, mode) as mk_fn:
    mk_fn.write(f'# {base_file_name}\n')
    mk_fn.write(f"## Train PCC Summary\n\n{train_pcc_markdown}\n\n")
    mk_fn.write(f"## Test PCC Summary\n\n{test_pcc_markdown}\n\n\n")

print(f"Train & Test PCC summary saved to {markdown_file_path}")
