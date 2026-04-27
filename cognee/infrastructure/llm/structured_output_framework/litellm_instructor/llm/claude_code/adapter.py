import asyncio
import json
import os
import shutil
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Resolve claude CLI path once at import time so misconfigured PATH surfaces early.
_CLAUDE_BIN = shutil.which("claude") or "claude"


class ClaudeCodeAdapter:
    """
    Routes LLM calls through the local `claude` CLI using the authenticated
    Max plan session. No ANTHROPIC_API_KEY required — auth is handled by the
    existing Claude Code login (OAuth / keychain).

    Usage (.env):
        LLM_PROVIDER=claude_code
        LLM_MODEL=claude-sonnet-4-6   # or 'sonnet', 'opus', etc.

    Uses asyncio.create_subprocess_exec (argument list, no shell=True)
    to avoid shell injection. The prompt text goes in as a positional arg,
    not interpolated into a shell string.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_completion_tokens: int = 16384,
        llm_args: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.llm_args = llm_args or {}

    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Call claude CLI with --json-schema for native structured output."""
        schema = response_model.model_json_schema()

        prompt = (
            f"System instructions: {system_prompt}\n\n"
            f"Extract information from the following input: {text_input}"
        )

        # All arguments passed as a list — no shell=True, no interpolation.
        # --setting-sources "" prevents the subprocess from loading user hooks
        # that would hang waiting for the parent Claude Code session infrastructure.
        cmd = [
            _CLAUDE_BIN,
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--json-schema", json.dumps(schema),
            "--no-session-persistence",
            "--setting-sources", "",
            prompt,
        ]

        # Inherit current env but signal hook infrastructure to stay inactive.
        env = {**os.environ, "DISABLE_OMC": "1"}

        logger.debug("ClaudeCodeAdapter invoking claude CLI (%s)", self.model)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {stderr.decode().strip()}"
            )

        envelope = json.loads(stdout.decode())

        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude CLI returned error: {envelope.get('result', 'unknown')}"
            )

        # With --json-schema, the validated object lives in envelope["structured_output"]
        # as a plain dict. Fall back to parsing envelope["result"] for older CLI versions.
        structured = envelope.get("structured_output")
        if structured is not None:
            return response_model.model_validate(structured)

        result_text = envelope["result"]
        try:
            return response_model.model_validate_json(result_text)
        except Exception:
            return response_model.model_validate(json.loads(result_text))

    async def create_transcript(self, input) -> str:
        raise NotImplementedError("Transcription not supported via Claude Code CLI.")

    async def transcribe_image(self, input) -> str:
        raise NotImplementedError("Image transcription not supported via Claude Code CLI.")
