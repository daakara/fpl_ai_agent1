import os
import shutil
from pathlib import Path

def organize_files():
    """Organizes files by moving unused files to the archive directory."""

    # Define the core files and directories
    core_files = [
        "core/simple_app.py",
        "core/main_app.py",
    ]

    # Define other important directories
    important_directories = [
        "components",
        "config",
        "utils",
        "data",
        "archive",
        "fpl_cache",
        "team_recommender",
        ".venv"
    ]

    # Create archive directory if it doesn't exist
    Path("archive").mkdir(parents=True, exist_ok=True)

    # Function to check if a file is a core file
    def is_core_file(file_path):
        return str(file_path).replace("\\","/") in core_files

    # Function to check if a path is an important directory
    def is_important_directory(path):
        return any(str(path).startswith(dir_path) for dir_path in important_directories)

    # Iterate through all files and directories in the project
    for item in Path(".").iterdir():
        if item.is_file() and not is_core_file(item) and str(item) != "organize_files.py" and str(item) != "run_cleanup.py" and str(item) != "create_clean_structure.py":
            try:
                shutil.move(str(item), "archive/" + str(item))
                print(f"Moved {item} to archive/")
            except Exception as e:
                print(f"Error moving {item}: {e}")
        elif item.is_dir() and not is_important_directory(item):
            try:
                shutil.move(str(item), "archive/" + str(item))
                print(f"Moved directory {item} to archive/")
            except Exception as e:
                print(f"Error moving directory {item}: {e}")

    print("File organization completed.")

if __name__ == "__main__":
    organize_files()