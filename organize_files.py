import os
import shutil
from pathlib import Path

def organize_files():
    """Organizes files by moving dependent files to appropriate directories and archiving unused files."""

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

    # Create necessary directories if they don't exist
    for dir_path in important_directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Function to check if a file is a core file
    def is_core_file(file_path):
        return str(file_path).replace("\\","/") in core_files

    # Function to check if a path is an important directory
    def is_important_directory(path):
        return any(str(path).startswith(dir_path) for dir_path in important_directories)

    # Move dependent files to appropriate directories
    dependent_files = {
        "components/data_loader.py": "components/data_loader.py",
        "components/fpl_official.py": "components/fpl_official.py",
        "components/llm_integration.py": "components/llm_integration.py",
        "components/team_recommender.py": "components/team_recommender.py",
        "utils/utils.py": "utils/utils.py",
        "config/config_template.py": "config/config_template.py",
    }

    for dest, src in dependent_files.items():
        try:
            shutil.move(src, dest)
            print(f"Moved {src} to {dest}")
        except FileNotFoundError:
            print(f"Warning: {src} not found.")
        except Exception as e:
            print(f"Error moving {src}: {e}")

    # Update import statements in core files
    def update_imports(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:  # Specify encoding
                content = f.read()

            # Update import statements based on new file locations
            content = content.replace("from data_loader", "from components.data_loader")
            content = content.replace("from fpl_official", "from components.fpl_official")
            content = content.replace("from llm_integration", "from components.llm_integration")
            content = content.replace("from team_recommender", "from components.team_recommender")
            content = content.replace("from utils", "from utils.utils")
            content = content.replace("from config_template", "from config.config_template")

            with open(file_path, "w", encoding="utf-8") as f:  # Specify encoding
                f.write(content)
            print(f"Updated imports in {file_path}")

        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError in {file_path}: {e}")
        except Exception as e:
            print(f"Error updating imports in {file_path}: {e}")

    for file_path in core_files:
        update_imports(file_path)

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