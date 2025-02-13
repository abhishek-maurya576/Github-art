import os
from datetime import datetime, timedelta

# Start from the first Sunday of the year
start_date = datetime(2022, 1, 2)  # Adjust for your target year

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

# Total number of weeks in a year (~52 weeks)
weeks_in_year = 52

# Adjust the gap to spread commits evenly
days_gap = (365 - 7 * len(pattern[0])) // len(pattern[0])  

# Loop through the pattern and commit on specific dates
for row in range(len(pattern)):  # Loop through weeks
    for col in range(len(pattern[row])):  # Loop through days in the week
        if pattern[row][col] == "#":
            commit_date = (start_date + timedelta(weeks=row, days=col * days_gap)).strftime("%Y-%m-%dT12:00:00")

            # Set environment variables for Git
            os.environ['GIT_AUTHOR_DATE'] = commit_date
            os.environ['GIT_COMMITTER_DATE'] = commit_date

            commit_message = f"Commit on {commit_date}"
            os.system(f'git commit --allow-empty -m "{commit_message}"')

# Push commits to GitHub
os.system("git push --force origin main")
