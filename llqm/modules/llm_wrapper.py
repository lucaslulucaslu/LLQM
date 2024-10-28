"""LLM Wrapper Module"""

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"
TEMPERATURE = 0


def chat_completion(sys_prompt, user_prompt, response_format=None):
    if response_format:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            temperature=TEMPERATURE,
        )
        response = completion.choices[0].message.parsed
    else:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )
        response = completion.choices[0].message.content

    return response
