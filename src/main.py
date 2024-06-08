from llms.togetherai import TogetherAILLM

async def main() -> None:
    llm = TogetherAILLM()
    res = await llm.arun("What is the capital of France?")
    print(res)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())