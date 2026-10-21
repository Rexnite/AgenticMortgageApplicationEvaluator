from core.llm import LLMClient

llm = LLMClient()

response = llm.chat(
    "You are a helpful assistant.",
    "Reply with exactly one word: Hello"
)

print(response)