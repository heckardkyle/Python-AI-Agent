import os
import sys
from dotenv import load_dotenv
from google import genai

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python3 main.py <prompt>")
        print("The prompt needs to be contained in quotations")
        print('ex:   python3 main.py "Tell me a joke"')
        sys.exit(1)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    prompt = sys.argv[1]

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=prompt
    )
    print(response.text)

    prompt_token_count = response.usage_metadata.prompt_token_count
    response_token_count = response.usage_metadata.candidates_token_count

    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Response tokens: {response_token_count}")

main()
