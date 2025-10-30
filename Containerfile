# Stage 1: Builder - Install dependencies in a temporary environment
# This stage keeps the final image clean and caches the dependency layer.
FROM python:3.13-slim AS builder

# Install system-level dependencies required by the application's shell tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
tree \
&& rm -rf /var/lib/apt/lists/*

# Set the working directory for the application.
WORKDIR /app

# Create a virtual environment to isolate dependencies.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the necessary files to install dependencies. This optimizes caching.
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install Python dependencies into the virtual environment.
# The '.' installs the project itself, including the 'agentic-tools' script.
RUN pip install --no-cache-dir .


# Stage 2: Final Image - Create the lean, production-ready image
FROM python:3.13-slim

# Set the working directory.
WORKDIR /app

# Create a non-root user for security. Running as a dedicated user is a critical
# security best practice, adhering to the principle of least privilege.
RUN useradd --system --create-home appuser

# Install only the necessary runtime system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
git \
tree \
&& rm -rf /var/lib/apt/lists/*

# Copy the virtual environment with installed dependencies from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the entire application source code into the final image.
COPY . .

# Set correct ownership for the application files.
RUN chown -R appuser:appuser /app

# Switch to the non-root user.
USER appuser

# Add the mounted volume to Git's safe directories to prevent 'dubious ownership' errors.
# This must be run as the non-root user.
RUN git config --global --add safe.directory /app

# Add the virtual environment's bin directory to the PATH.
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Set the entry point to the installed script. This makes the container
# behave like a standalone executable.
ENTRYPOINT ["agentic-tools"]

# Set a default command to display help, making the container self-documenting.
CMD ["--help"]