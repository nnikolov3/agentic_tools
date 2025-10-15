"""
File: src/approver.py
Author: Niko Nikolov
Scope: Implements the logic for the Approver agent.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, cast

# Internal imports
from src._api import api_caller
from src.shell_tools import collect_documentation, collect_sources

logger = logging.getLogger(__name__)


class Approver:
    """
    An agent that makes a final decision on code changes by reviewing context
    from design documents, and the code itself.
    """

    def __init__(self, approver_configuration: Dict[str, Any]) -> None:
        """
        Initializes the Approver agent with its configuration.

        Parameters:
            approver_configuration (Dict[str, Any]): The configuration dictionary
                for the approver agent from the TOML file.
        """
        if not isinstance(approver_configuration, dict):
            raise ValueError("approver_configuration must be a dict")

        self.configuration = approver_configuration

        # Handle project_root, replacing "PWD" with the current working directory
        project_root_path = self.configuration.get("project_root", ".")
        if project_root_path == "PWD":
            project_root_path = os.getcwd()

        self.project_root = project_root_path

    def _assemble_payload(
        self,
        docs_context: str,
        code_context: str,
        user_chat: str,
    ) -> str:
        """
        Assembles the complete, detailed payload to be sent to the LLM.

        Parameters:
            docs_context (str): Content of all project documentation files.
            code_context (str): Content of all recently modified source files.
            user_chat (str): The recent user chat history.

        Returns:
            str: The fully assembled prompt payload.
        """
        return f"""# Full Context for Final Review

        ## 1. Design Documents & Principles:
        {docs_context}

        ## 2. Recent Code Changes:
        {code_context}

        ## 3. User Chat History:
        {user_chat}

        ## Your Task:
        Based on all the context provided above, perform a final review and provide your verdict.
        """

    def _parse_and_validate_json(self, json_string: str) -> Optional[Dict[str, Any]]:
        """
        Parses the JSON string from the LLM and validates its structure.

        Parameters:
            json_string (str): The JSON string returned by the LLM.

        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary if valid, otherwise None.
        """
        try:
            data: Any = json.loads(json_string)

            required_keys = [
                "decision",
                "summary",
                "positive_points",
                "negative_points",
                "required_actions",
            ]

            if isinstance(data, dict) and all(key in data for key in required_keys):
                return cast(Dict[str, Any], data)
            else:
                logger.error(
                    f"LLM response JSON is missing required keys. Response: {data}"
                )
                return None
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode LLM response as JSON. Response: {json_string}"
            )
            return None

    def execute(self, user_chat: str) -> Dict[str, Any]:
        """
        The main entry point for the agent. It gathers context, calls the LLM,
        and returns a structured verdict.

        Parameters:
            user_chat (str): The recent user chat history.

        Returns:
            Dict[str, Any]: A dictionary containing the structured verdict.
        """
        logger.info("Approver agent started. Gathering context...")
        docs_context = collect_documentation(self.project_root)
        code_context = collect_sources(self.project_root)

        full_payload = self._assemble_payload(docs_context, code_context, user_chat)

        logger.info("Approver agent making decision...")
        llm_response = api_caller(self.configuration, full_payload)

        error_verdict = {
            "decision": "ERROR",
            "summary": "Agent failed to get a valid response from the LLM.",
            "positive_points": [],
            "negative_points": ["The LLM response was missing or malformed."],
            "required_actions": ["Debug the connection to the LLM provider."],
        }

        if llm_response and llm_response.content:
            verdict = self._parse_and_validate_json(llm_response.content)
            return verdict if verdict is not None else error_verdict
        else:
            return error_verdict
