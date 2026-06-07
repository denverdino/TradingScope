# -*- coding: utf-8 -*-
"""Qwen Deep Research Agent for AgentScope 2.0.

Calls the qwen-deep-research model directly via DashScope streaming API.
Supports a two-step research process: clarification → deep research.
"""

import os
from typing import Optional

import dashscope
from agentscope import logger
from agentscope.message import Msg, UserMsg
from dashscope.api_entities.dashscope_response import GenerationResponse


class QwenDeepResearchAgent:
    """Deep Research Agent based on the Qwen-Deep-Research model.

    This is a standalone agent (not an AgentScope Agent subclass) because it
    calls the DashScope streaming API directly rather than going through
    AgentScope's model layer.
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.name = name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "The DASHSCOPE_API_KEY environment variable is not set.",
            )
        self.model_name = "qwen-deep-research"
        self.verbose = verbose
        self._messages: list[dict[str, str]] = []

    async def __call__(self, x: Msg) -> Msg:
        """Process an input message and return a reply."""
        self._messages.append({"role": x.role, "content": x.get_text_content()})

        user_count = sum(1 for m in self._messages if m["role"] == "user")
        step_name = "Clarification" if user_count == 1 else "Deep Research"
        logger.info("[%s] Starting %s ...", self.name, step_name)

        content = await self._call_model(step_name)

        self._messages.append({"role": "assistant", "content": content})

        return UserMsg(name=self.name, content=content)

    async def _call_model(self, step_name: str) -> str:
        if self.verbose:
            logger.info("\n%s", "=" * 50)
            logger.info("  %s", step_name)
            logger.info("%s", "=" * 50)

        try:
            responses = await dashscope.AioGeneration.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=self._messages,
                stream=True,
                request_timeout=1800,
            )
            return await self._process_responses(responses)
        except Exception as e:
            err_msg = f"An error occurred when calling the API: {e}"
            logger.error(err_msg)
            return err_msg

    async def _process_responses(
        self,
        responses: GenerationResponse,
    ) -> str:
        current_phase = None
        current_status = None
        phase_content = ""
        research_goal = ""
        keepalive_shown = False
        references = []

        async for response in responses:
            if hasattr(response, "status_code") and response.status_code != 200:
                error_msg = f"HTTP status code: {response.status_code}"
                if hasattr(response, "code"):
                    error_msg += f", Error code: {response.code}"
                if hasattr(response, "message"):
                    error_msg += f", Error message: {response.message}"
                logger.error(error_msg)
                continue

            if hasattr(response, "output") and response.output:
                message = response.output.get("message", {})
                phase = message.get("phase")
                content = message.get("content", "")
                status = message.get("status")
                extra = message.get("extra", {})

                if phase != current_phase:
                    if current_phase and phase_content and self.verbose:
                        logger.info("\n✓ %s phase completed", current_phase)

                    current_phase = phase
                    phase_content = ""
                    keepalive_shown = False

                    if phase and phase != "KeepAlive" and self.verbose:
                        logger.info("\n▶ Entering %s phase", phase)
                        if phase == "answer":
                            references = extra.get("deep_research", {}).get(
                                "references",
                                [],
                            )

                if phase == "WebResearch" and self.verbose:
                    research_goal = self._handle_web_research_phase(
                        status,
                        extra,
                        research_goal,
                    )

                if content:
                    phase_content += content
                    if self.verbose:
                        logger.debug(content)

                if status:
                    if status != current_status and status != "typing" and self.verbose:
                        self._log_status(status)
                    current_status = status

                if status == "finished":
                    self._log_usage(response)
                    if self.verbose:
                        logger.info("\n✓ %s phase completed", current_phase)
                    if phase == "answer":
                        if len(references) > 0:
                            reference_links = []
                            list_links = []
                            for i, ref in enumerate(references):
                                title = ref["title"]
                                url = ref["url"]
                                reference_links.append(
                                    f'[{i + 1}]: {url} "{title}"',
                                )
                                list_links.append(f"{i + 1}. [{title}]({url})")
                            phase_content = phase_content + "\n\n## References\n\n" + "\n".join(list_links) + "\n\n" + "\n".join(reference_links)
                            break

                if phase == "KeepAlive":
                    if not keepalive_shown and self.verbose:
                        logger.info("\n⏳ Preparing for the next phase...")
                        keepalive_shown = True
                    continue
        return phase_content

    def _handle_web_research_phase(
        self,
        status: str,
        extra: dict,
        research_goal: str,
    ) -> str:
        web_sites = []
        if extra.get("deep_research", {}).get("research"):
            research_info = extra["deep_research"]["research"]

            if status == "streamingQueries":
                if "researchGoal" in research_info:
                    goal = research_info["researchGoal"]
                    if goal:
                        research_goal += goal

            elif status == "streamingWebResult":
                if research_goal != "":
                    logger.info("\n🎯 Research Goal: %s", research_goal)
                    research_goal = ""
                if "webSites" in research_info:
                    sites = research_info["webSites"]
                    if sites and sites != web_sites:
                        web_sites.extend(sites)
                        msg = f"\n🔍 Found {len(sites)} relevant websites:\n" + "\n".join(
                            f"  {i + 1}. {site.get('title', 'No title')}\n     {site.get('url', 'No link')}" for i, site in enumerate(sites)
                        )
                        logger.info(msg)
            elif status == "WebResultFinished":
                logger.info(
                    "\n✓ Web search completed, found %s reference sources",
                    len(web_sites),
                )

        return research_goal

    def _log_status(self, status: str) -> None:
        status_desc = {
            "streamingQueries": "Generating research goals and search queries (WebResearch phase)",
            "streamingWebResult": "Performing search, web page reading, and code execution (WebResearch phase)",
            "WebResultFinished": "Web search phase completed (WebResearch phase)",
        }
        if status in status_desc:
            logger.info("\n📊 %s", status_desc[status])

    def _log_usage(self, response: GenerationResponse) -> None:
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            if self.verbose:
                logger.info(
                    "\n📈 Token usage - input: %s output: %s",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )

    async def reset_memory(self) -> None:
        self._messages.clear()
