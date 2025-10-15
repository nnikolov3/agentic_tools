"""
Provides essential shell tools to an agent.
"""

import logging

import subprocess



logger = logging.getLogger(__name__)





def collect_sources(project_root):

    """Finds recently modified files and returns their content as a single string."""

    modified_sources = ""

    try:

        # Find files modified in the last 5 minutes

        result = subprocess.run(

            ["find", project_root, "-type", "f", "-mmin", "-5"],

            capture_output=True,

            text=True,

            check=True,

            timeout=30,  # Add a 30-second timeout

        )

        file_paths = result.stdout.strip().split("\n")



        for file_path in file_paths:

            if file_path:

                try:

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                        file_content = f.read()

                        modified_sources += file_content + "\n"

                except (FileNotFoundError, IOError) as e:

                    logger.error(f"Error reading file {file_path}: {e}")



    except subprocess.CalledProcessError as e:

        logger.error(f"Error finding files: {e}")

    except (FileNotFoundError, IOError) as e:

        logger.error(f"An unexpected error occurred: {e}")

    except subprocess.TimeoutExpired:

        logger.warning("collect_sources: find command timed out.")



    return modified_sources





def collect_documentation(project_root):

    """Finds recently modified files and returns their content as a single string."""

    documentation = ""

    try:

        # Find files modified in the last 5 minutes

        result = subprocess.run(

            ["find", project_root, "-type", "f", "-name", "*.md", "-mmin", "-5"],

            capture_output=True,

            text=True,

            check=True,

            timeout=30,  # Add a 30-second timeout

        )

        file_paths = result.stdout.strip().split("\n")



        for file_path in file_paths:

            if file_path:

                try:

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                        file_content = f.read()

                        documentation += file_content + "\n"

                except (FileNotFoundError, IOError) as e:

                    logger.error(f"Error reading file {file_path}: {e}")



    except subprocess.CalledProcessError as e:

        logger.error(f"Error finding files: {e}")

    except (FileNotFoundError, IOError) as e:

        logger.error(f"An unexpected error occurred: {e}")

    except subprocess.TimeoutExpired:

        logger.warning("collect_documentation: find command timed out.")



    return documentation
