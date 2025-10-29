# --- Stage 1: Builder ---
# Use a full Python image that includes build tools to compile dependencies.
FROM python:3.13-bookworm AS builder

# Set the working directory
WORKDIR /app

# Create a virtual environment for clean dependency management
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the files needed to install dependencies
# This leverages Podman's layer caching. Dependencies are only re-installed
# if these files change.
COPY pyproject.toml README.md MANIFEST.in ./

# Install the project dependencies. This also installs the project in editable
# mode, making the `agentic-tools` script available.
RUN pip install --no-cache-dir .

# --- Stage 2: Final Image ---
# Use a slim base image for the final application to reduce size.
FROM python:3.13-slim-bookworm

# Create a non-root user and group for security
RUN groupadd --system app && useradd --system --gid app app
USER app
WORKDIR /home/app

# Copy the virtual environment with installed dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY --chown=app:app src/ ./src
COPY --chown=app:app main.py ./

# Set the PATH to include the virtual environment's binaries
ENV PATH="/opt/venv/bin:$PATH"

# The entrypoint is the script defined in pyproject.toml.
# This makes the container behave like the command-line tool.
ENTRYPOINT ["agentic-tools"]

# Default command if no arguments are provided to `podman run`.
# By default, it will show the help message.
CMD ["--help"]
