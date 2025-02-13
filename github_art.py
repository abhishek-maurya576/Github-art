import os
from datetime import datetime, timedelta

# Start date (change to a Sunday for better alignment)
start_date = datetime(2022, 1, 2)  # Adjust to your desired year

# "I LOVE YOU" pattern
pattern = [
    "  ##   ###  ###   ###  #   #  ###   ###  ",
    "  ##  #   # #  # #     #   #  #  # #   # ",
    "  ##  #   # #  # #     #   #  #  # #   # ",
    "  ##  #   # ###   ##   #####  ###   ##   ",
    "      #   # #  #    #  #   #  # #     #  ",
    "      #   # #  #    #  #   #  #  #    #  ",
    "       ###  ###  ###   #   #  #   # ###  "
]

# Loop through the pattern and commit on specific dates
for row in range(len(pattern)):
    for col in range(len(pattern[row])):
        if pattern[row][col] == "#":
            commit_date = (start_date + timedelta(days=row*7 + col)).strftime("%Y-%m-%dT12:00:00")  # Set a fixed time

            # Set environment variables for Git
            os.environ['GIT_AUTHOR_DATE'] = commit_date
            os.environ['GIT_COMMITTER_DATE'] = commit_date

            commit_message = f"Commit on {commit_date}"
            os.system("git commit --allow-empty -m \"" + commit_message + "\"")

# Push commits to GitHub
os.system("git push origin main")
