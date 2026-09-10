import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

messages = []

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:fastest",
        messages=messages
    )

    answer = response.choices[0].message.content

    print("AI:", answer)

    messages.append({
        "role": "assistant",
        "content": answer
    })