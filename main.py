import os

import sys

from prompts import system_prompt

from dotenv import load_dotenv

from openai import OpenAI

import argparse

from call_function import available_functions

from call_function import call_function

import json

def main():

    parser = argparse.ArgumentParser(description="Chatbot")

    parser.add_argument("user_prompt", type=str, help="User prompt")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    messages = [

        {"role": "system", "content": system_prompt},

        {"role": "user", "content": args.user_prompt},
    ]

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key == None:
        raise RuntimeError("api_key not found")

    client = OpenAI(

        base_url="https://openrouter.ai/api/v1",

        api_key=api_key,

    )

    for n in range(20):

        response = client.chat.completions.create(

            model="openrouter/free",

            messages= messages,

            tools=available_functions,

        )

        message = response.choices[0].message

        messages.append(message)

        if response.usage == None:
            raise RuntimeError("failed API request")

        if message.tool_calls:

            for tool_call in message.tool_calls:

                result_message = call_function(tool_call, args.verbose)

                if not result_message['content']:

                    raise Exception("Error")

                messages.append(result_message)

                if args.verbose:

                    print(f"-> {result_message['content']}")

        else:

            if args.verbose == True:
                print(f"User prompt: {args.user_prompt}")
                print(f"Prompt tokens: {response.usage.prompt_tokens}")
                print(f"Response tokens: {response.usage.completion_tokens}")

            print(response.choices[0].message.content)

            break

    if n == 20:

        print("maximum iterations reached without response")

        sys.exit(1)

if __name__ == "__main__":
    main()
