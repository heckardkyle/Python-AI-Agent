import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():
    valid_flags = [
        "--verbose"
    ]

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 main.py <prompt>")
        print("The prompt needs to be contained in quotations")
        print('ex:   python3 main.py "Tell me a joke"')
        sys.exit(1)

    used_flags = []

    for i in range (2, len(sys.argv)): 
        if sys.argv[i] not in valid_flags:
            print(f"{sys.argv[i]} is not a valid argument")
            print("Valid flags:")
            for flag in valid_flags:
                print(f"    {flag}")
                sys.exit(1)
        used_flags.append(sys.argv[i])

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    user_prompt = sys.argv[1]
    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]  

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=messages
    )

    prompt_tokens = response.usage_metadata.prompt_token_count
    response_tokens = response.usage_metadata.candidates_token_count

    if "--verbose" in used_flags:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Response tokens: {response_tokens}")

    print(response.text)

main()
