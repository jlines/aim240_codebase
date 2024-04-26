import pandas as pd

# Specify the file path of the CSV file
file_path = "data/Jira.csv"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# get a list of employee id
employees = df["Watchers Id"].drop_duplicates().dropna().to_list()

# get the comment columns
mask = df.columns.str.contains("Comment.*")
clean_df = df.loc[:, mask]

# get the saved columns
saved_cols = ["Summary", "Created", "Issue key", "Resolution", "Description"]
clean_df.loc[:, saved_cols] = df.loc[:, saved_cols]

# make the description column consistent with the comment columns
clean_df.loc[:, "Description"] = (
    clean_df["Created"].astype(str) + ";xxxxx;" + clean_df["Description"].astype(str)
)

# pivot the comment columns into rows
meltcols = ["Summary", "Issue key", "Resolution"]
mdf = clean_df.melt(id_vars=meltcols, var_name="temp", value_name="Message").drop(
    columns="temp"
)

# remove null comments
mdf = mdf[mdf["Message"].notna()]

# Split the comment metadata into separate columns
mdf[["Date", "UserId", "Message"]] = mdf["Message"].str.split(";", n=2, expand=True)

# Mark messages as customer messages or support responses
mdf["SupportResponse"] = mdf["UserId"].isin(employees)
mdf.drop(columns=["UserId"], inplace=True)

# Save the cleaned data to a new CSV file
mdf.to_csv("data/CleanedJira.csv", index=False)
