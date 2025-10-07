#!/bin/bash
# install_gemini.sh - Install MCP server to Gemini CLI

set -e

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading .env..."
    set -a
    source .env
    set +a
else
    echo "❌ .env file not found!"
    exit 1
fi

echo "🔧 Installing Multi-Agent Development Team to Gemini CLI..."

# Build env string for gemini mcp add
ENV_FLAGS=""
for key in GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY GROQ_API_KEY \
    CEREBRAS_API_KEY_PERSONAL CEREBRAS_API_KEY_BOOK_EXPERT \
    DASHSCOPE_API_KEY QDRANT_URL QDRANT_API_KEY; do
    value="${!key}"
    if [ -n "$value" ]; then
        ENV_FLAGS="$ENV_FLAGS --env $key=$value"
    fi
done

# Add to gemini CLI
gemini mcp add multi-agent-dev-team \
    "fastmcp run $(pwd)/multi_agent_dev_team.py" \
    --transport stdio \
    --description "Multi-agent development team with intelligent routing and Qdrant memory" \
    $ENV_FLAGS

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Start gemini CLI:"
echo "   gemini"
echo ""
echo "📊 Test the server:"
echo "   > Use list_team_members"
echo ""
