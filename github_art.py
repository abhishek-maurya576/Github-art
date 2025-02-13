import os
from datetime import datetime, timedelta

# Start from the first Sunday of the year for better alignment
start_date = datetime(2022, 1, 2)  # Adjust to match your desired year

# "I LOVE YOU" pattern (7 rows for weeks, spread across 12 months)
pattern = [
    "  ##   ###  ###   ###  #   #  ###   ###  ",
    "  ##  #   # #  # #     #   #  #  # #   # ",
    "  ##  #   # #  # #     #   #  #  # #   # ",
    "  ##  #   # ###   ##   #####  ###   ##   ",
    "      #   # #  #    #  #   #  # #     #  ",
    "      #   # #  #    #  #   #  #  #    #  ",
    "       ###  ###  ###   #   #  #   # ###  "
]

# Ensure commits are distributed throughout the year
days_gap = 365 // len(pattern[0])  # Space commits throughout the year

# Loop through the pattern and commit on specific dates
for row in range(len(pattern)):
    for col in range(len(pattern[row])):
        if pattern[row][col] == "#":
            commit_date = (start_date + timedelta(days=row*7 + col * days_gap)).strftime("%Y-%m-%dT12:00:00")

            # Set environment variables for Git
            os.environ['GIT_AUTHOR_DATE'] = commit_date
            os.environ['GIT_COMMITTER_DATE'] = commit_date

            commit_message = f"Commit on {commit_date}"
            os.system(f'git commit --allow-empty -m "{commit_message}"')

# Push commits to GitHub
os.system("git push --force origin main")
