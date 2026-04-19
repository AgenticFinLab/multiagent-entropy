"""Constants for GAIA benchmark experiments."""

GAIA_TASK_TYPE = "gaia"
GAIA_DATA_PATH = "experiments/data/GAIA"
GAIA_SPLIT = "validation"

# Root directory of downloaded GAIA attachments (from snapshot_download).
# After running scripts/download_gaia_attachments.py, attachments are stored under
# <GAIA_ATTACHMENTS_ROOT>/<file_path>, where file_path is the value from sample_info.
# e.g. experiments/data/GAIA/attachments/2023/validation/xxx.xlsx
GAIA_ATTACHMENTS_ROOT = "experiments/data/GAIA/attachments"

# Official GAIA system prompt (from the GAIA leaderboard / paper)
GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. "
    "Report your thoughts, and finish your answer with the following template: "
    "FINAL ANSWER: [YOUR FINAL ANSWER].\n"
    "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.\n"
    "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.\n"
    "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
    "and write the digits in plain text unless specified otherwise.\n"
    "If you are asked for a comma separated list, apply the above rules depending of whether the "
    "element to be put in the list is a number or a string."
)
