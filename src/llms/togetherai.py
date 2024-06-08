from typing import AsyncGenerator
from src.prompts.system_messages import ASSISTANT
from src.llms.base import BaseLLM, T
from src.configs import togetherai_client, structured_togetherai_client


class TogetherAILLM(BaseLLM):
    """
    Initializes a new instance of the TogetherAILLM class.

    Args:
        prompt (str, optional): The prompt to use for the AI model. Defaults to ASSISTANT.
        model (str, optional): The name of the AI model to use. Defaults to "mistralai/Mistral-7B-Instruct-v0.1".

    Returns:
        None
    """

    def __init__(
        self,
        prompt: str = ASSISTANT,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
    ) -> None:
        super().__init__(model=model, prompt=prompt)

    async def arun(self, text: str) -> str:
        response = await togetherai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )

        return response.choices[0].message.content or ""

    async def astream(self, text: str) -> AsyncGenerator[str, None]:
        stream = await togetherai_client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": text}], stream=True
        )

        async for chunk in stream:
            delta_content = chunk.choices[0].delta.content or ""
            yield delta_content

    async def astructured_extraction(self, text: str, output_class: type[T]) -> T:
        response = await structured_togetherai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Let's think step by step. Given a text, extract structured data from it.",
                },
                {"role": "user", "content": text},
            ],
            max_retries=3,
            response_model=output_class,
        )
        return response
