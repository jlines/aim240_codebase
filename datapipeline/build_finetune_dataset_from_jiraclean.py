import pandas as pd

# Specify the path to your CSV file
csv_file = "data/CleanedJira.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file)

columns = ["Summary", "IssueKey", "Resolution", "Message", "Date", "SupportResponse"]

# Assign the columns to the DataFrame
df.columns = columns

# Sort the dataframe by IssueKey and Date
df = df.sort_values(by=["IssueKey", "Date"])

# Drop the rows with empty values in the message column
df = df.dropna(subset=["Message"])

# remove strings from the message column that are contained in []
df["Message"] = df["Message"].str.replace(r"\[.*\]", "", regex=True)

# replace all newlines with a space
df["Message"] = df["Message"].str.replace(r"\n", " ", regex=True)
df["Message"] = df["Message"].str.replace(r"\r", "", regex=True)
df["Message"] = df["Message"].str.replace(r"  ", " ", regex=True)

# open file for writing
f = open("data/finetune_dataset.txt", "w")

current_issue = None
current_datapoint = ""

for line in df.iterrows():
    source = "Human"
    if line[1]["SupportResponse"] is True:
        source = "Agent"

    if current_issue != line[1]["IssueKey"]:
        if current_issue is not None:
            f.write(current_datapoint + "\n")

        current_issue = line[1]["IssueKey"]
        current_datapoint = f"### {source}: {line[1]['Summary']} {line[1]['Message']} "

    else:
        current_datapoint += f"### {source}: {line[1]['Message']} "
