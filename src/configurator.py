import tomllib
from pathlib import Path


class Configurator:
    def __init__(self, config_path):
        self.config_path = config_path  # This is a string path like "config.toml"

    def get_config_dictionary(self):
        try:
            # Validate path exists
            if not Path(self.config_path).exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            # Validate it's a file, not a directory
            if not Path(self.config_path).is_file():
                raise IsADirectoryError(
                    f"Path is a directory, not a file: {self.config_path}"
                )

            # Open and parse TOML file
            with open(self.config_path, "rb") as f:

                configuration = tomllib.load(f)

                return configuration

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied reading config file: {self.config_path}"
            ) from e
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML syntax in config file: {e}") from e
        except OSError as e:
            raise OSError(f"Error reading config file {self.config_path}: {e}") from e
