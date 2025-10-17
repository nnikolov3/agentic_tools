# File: src/approver.py
"""
Approver agent: assembles context and requests a final decision as strict JSON.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.base_agent import BaseAgent, BaseInputs
from src.qdrant_integration import QdrantIntegration


logger = logging.getLogger(__name__)


def store_decision(
    qdrant_integration: QdrantIntegration,
    storage_id: str,
    content_for_embedding: str,
    decision_data: Dict[str, Any],
) -> bool:
    """
    Performs the actual storage of the approver decision in Qdrant.

    Args:
        qdrant_integration: The initialized QdrantIntegration object.
        storage_id: The unique ID for the storage point.
        content_for_embedding: The text content used to generate the vector embedding.
        decision_data: The dictionary containing the parsed LLM decision.

    Returns:
        True if storage was successful, False otherwise.
    """
    decision_with_timestamp = dict(decision_data)
    decision_with_timestamp["timestamp"] = time.time()
    return bool(
        qdrant_integration.store_approver_decision(
            decision_with_timestamp, storage_id, content_for_embedding
        )
    )


class Approver(BaseAgent):
    """
    Final gatekeeper that reads design docs and recent code changes, then returns a JSON decision.
    """

    def get_agent_name(self) -> str:
        return "approver"

    def _create_messages(
        self, inputs: BaseInputs, docs: str, sources: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        # Extract user_chat from context
        user_chat = context.get("user_chat", "") if context else ""

        # Get skills from agent config if available
        agent_config = self._configurator.get_agent_config("approver")
        skills = agent_config.get("skills", [])

        # Use the configurator's method to combine prompt with skills
        system = self._configurator.combine_prompt_with_skills(
            inputs.prompt.strip(), tuple(skills)
        )

        user = (
            "# Full Context for Final Review\n\n"
            "## 1. Design Documents & Principles:\n"
            f"{docs or '(no docs found)'}\n\n"
            "## 2. Recent Code Changes:\n"
            f"{sources or '(no recent source changes found)'}\n\n"
            "## 3. User Chat History:\n"
            f"{user_chat.strip()}\n\n"
            "## Your Task:\n"
            "Return ONLY the required JSON object with the specified schema.\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _extract_approver_content_for_embedding(self, data: Dict[str, Any]) -> str:
        """
        Extract content from decision data for embedding.

        Args:
            data: The decision data

        Returns:
            Content string for embedding
        """
        summary = data.get("summary", "")
        positive_points = "; ".join(data.get("positive_points", []))
        negative_points = "; ".join(data.get("negative_points", []))
        required_actions = "; ".join(data.get("required_actions", []))
        decision_type = data.get("decision", "")
        return f"Decision: {decision_type}, Summary: {summary}, Positive: {positive_points}, Negative: {negative_points}, Actions: {required_actions}"

    def _store_approver_decision_in_qdrant(self, decision_data: Dict[str, Any]) -> None:
        """
        Store approver decision in Qdrant if enabled.

        Args:
            decision_data: The decision data to store
        """
        self._store_in_qdrant_if_enabled(
            agent_name=self.get_agent_name(),
            data=decision_data,
            store_func=store_decision,
            content_extractor_func=self._extract_approver_content_for_embedding,
        )

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self._load_inputs()
        docs = self._assemble_docs(inputs)

        # Unpack the new return value: sources content and all candidate files
        sources, candidate_files = self._assemble_sources(inputs)

        if not sources:
            # Use the candidate_files list returned by collect_recent_sources
            # The candidate files are already Path objects, so we convert them to relative strings
            src_files = [
                str(candidate_file.relative_to(Path(inputs.project_root)))
                for candidate_file in candidate_files
            ]

            return {
                "status": "no_recent_files",
                "data": {"files": src_files},
                "message": "No recent source files were found to analyze. Please choose a file from the list below and re-run the tool with the file path as an argument.",
            }

        user_chat = payload.get("user_chat", "")
        context = {"user_chat": user_chat}
        messages = self._create_messages(inputs, docs, sources, context)
        result = self._make_api_call(inputs, messages)

        # Check if the API call was successful and contains decision data
        if result.get("status") == "success":
            raw_response = result["data"].get("raw_text", "")

            # Clean the raw response by removing markdown fences if present
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```"):
                # Find the start and end of the JSON object
                start_index = cleaned_response.find('{')
                end_index = cleaned_response.rfind('}')
                if start_index != -1 and end_index != -1:
                    cleaned_response = cleaned_response[start_index:end_index+1]

            try:
                # Attempt to parse the response as JSON (the approver returns strict JSON)
                decision_data = json.loads(cleaned_response)

                # Check if the decision is APPROVED and that the result is a dictionary
                if (
                    isinstance(decision_data, dict)
                    and decision_data.get("decision") == "APPROVED"
                ):
                    # Add user chat to decision data
                    decision_data["user_chat"] = user_chat
                    self._store_approver_decision_in_qdrant(decision_data)

            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse raw_text as JSON, skipping Qdrant storage"
                )
            except Exception as processing_error:
                logger.error(
                    f"Error processing decision for Qdrant storage: {processing_error}"
                )

        return result
