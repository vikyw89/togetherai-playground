# llm is a class that takes a text and returns a response text
from abc import abstractmethod
from typing import AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_incrementing
from src.prompts.system_messages import ASSISTANT
from together.types import ChatCompletionChunk


class BaseLLM:
    def __init__(
        self,
        model: str,
        prompt: str = ASSISTANT,
        verbose: bool = False,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.verbose = verbose

    @retry(stop=stop_after_attempt(3), wait=wait_incrementing(start=1, increment=1))
    async def arun(self, text: str) -> str:
        return await self._arun(text)

    @retry(stop=stop_after_attempt(3), wait=wait_incrementing(start=1, increment=1))
    async def astream(self, text: str) -> AsyncGenerator[str, None]:
        stream = await self._astream(text)
        async for chunk in stream:
            if chunk.choices is None:
                continue
            if chunk.choices[0].delta is None:
                continue
            chunk_message = chunk.choices[0].delta.content  # extract the message
            yield chunk_message or ""

    @abstractmethod
    async def _arun(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def _astream(self, text: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        raise NotImplementedError
