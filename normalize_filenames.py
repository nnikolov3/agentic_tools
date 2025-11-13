import os
import re

def normalize_filename(filename: str) -> str:
    """
    Normalizes a filename by converting to lowercase, removing special
    characters, and replacing spaces/hyphens with underscores.
    """
    # Separate the name and extension
    name, extension = os.path.splitext(filename)
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters, keeping only letters, numbers, and spaces
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s-]+', '_', name)
    
    # Remove any leading/trailing underscores
    name = name.strip('_')
    
    return f"{name}{extension}"

def main():
    """
    Main function to rename all PDF files in the current directory.
    """
    current_directory = os.getcwd()
    
    for filename in os.listdir(current_directory):
        if filename.lower().endswith('.pdf'):
            normalized_name = normalize_filename(filename)
            
            if filename != normalized_name:
                try:
                    old_path = os.path.join(current_directory, filename)
                    new_path = os.path.join(current_directory, normalized_name)
                    
                    os.rename(old_path, new_path)
                    print(f'Renamed: "{filename}" -> "{normalized_name}"')
                except OSError as e:
                    print(f'Error renaming "{filename}": {e}')

if __name__ == "__main__":
    main()
