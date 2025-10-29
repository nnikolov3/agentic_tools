import asyncio
import main
from pathlib import Path

async def run_approver():
    with open("prompt.txt", "r") as f:
        chat_prompt = f.read()
    main.configuration = main._load_configuration(main.CONFIG_FILE_PATH)
    await main._run_agent_tool("approver", chat_prompt, None, Path.cwd())

if __name__ == "__main__":
    asyncio.run(run_approver())
