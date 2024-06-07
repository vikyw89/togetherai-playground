from typing import AsyncGenerator
from src.prompts.system_messages import ASSISTANT
from src.llms.base import BaseLLM
from src.config import client
from together.types import ChatCompletionResponse, ChatCompletionChunk


class TogetherAILLM(BaseLLM):
    def __init__(
        self,
        prompt: str = ASSISTANT,
        model: str = "Qwen/Qwen1.5-0.5B-Chat",
    ) -> None:
        super().__init__(model=model, prompt=prompt)

    async def _arun(self, text: str) -> str | None:
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )

        if isinstance(response, ChatCompletionResponse):
            choices = response.choices

            if choices is None:
                return None

            if choices[0].message is None:
                return None

            return choices[0].message.content

    async def _astream(self, text: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            stream=True,
        )

        if not isinstance(stream, AsyncGenerator):
            raise TypeError("Stream is not an async generator")

        return stream
