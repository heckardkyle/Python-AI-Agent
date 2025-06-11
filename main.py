import os
import sys
from dotenv import load_dotenv
from google import genai
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_python_file import run_python_file
from functions.write_file import write_file
from google.genai import types

def call_function(function_call_part, verbose=False):
    functions = {
        "get_files_info": get_files_info,
        "get_file_content": get_file_content,
        "write_file": write_file,
        "run_python_file": run_python_file,
    }
    if verbose:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_call_part.name}")

    if function_call_part.name not in functions:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_call_part.name,
                    response={"error": f"Unknown function: {function_call_part.name}"},
                )
            ],
        )

    result = functions[function_call_part.name]("./calculator", **function_call_part.args)

    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_call_part.name,
                response={"result": result},
            )
        ],
    )

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

    schema_get_files_info = types.FunctionDeclaration(
        name="get_files_info",
        description="Lists files in the specified directory along with their sizes, constrained to the working directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "directory": types.Schema(
                    type=types.Type.STRING,
                    description="The directory to list files from, relative to the working directory. If not provided, lists files in the working directory itself.",
                ),
            },
        ),
    )

    schema_get_file_content = types.FunctionDeclaration(
        name="get_file_content",
        description="Reads the Contents of a file.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="The file path to read contents from, relative to the working directory. If not provided, reads a file in the working directory itself.",
                ),
            },
        ),
    )
    
    schema_write_file = types.FunctionDeclaration(
        name="write_file",
        description="Overwrites an existing file.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="The file path of the file to write content to, relative to the working directory. If not provided, write to a file in the working directory itself.",
                ),
                "content": types.Schema(
                    type=types.Type.STRING,
                    description="The content to write to a file.",
                ),
            },
        ),
    )

    schema_run_python_file = types.FunctionDeclaration(
        name="run_python_file",
        description="Runs a python file.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="The file path of the python file to run, relative to the working directory. If not provided, lists files in the working directory itself.",
                ),
                "args": types.Schema(
                    type=types.Type.STRING,
                    description="Defaults to None. Enter required or optional arguments for the python file here.",
                ),
            },
        ),
    )

    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_get_file_content,
            schema_write_file,
            schema_run_python_file,
        ]
    )

    system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""

    user_prompt = sys.argv[1]
    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    for i in range(20):

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=messages,
            config=types.GenerateContentConfig(
                tools=[available_functions],
                system_instruction = system_prompt
            ),
        )

        for candidate in response.candidates:
            messages.append(candidate.content)

        if response.function_calls:
            for function_call in response.function_calls:
                if "--verbose" in used_flags:
                    function_call_result = call_function(function_call, verbose=True)
                else:
                    function_call_result = call_function(function_call)
                messages.append(function_call_result)
                if not function_call_result.parts[0].function_response.response:
                    raise Exception("Fatal Exception. No valid response.")
                else:
                    if "--verbose" in used_flags:
                        print(f"-> {function_call_result.parts[0].function_response.response}")

        if not response.function_calls or i == 19:
            print(response.text)
            break
        


    prompt_tokens = response.usage_metadata.prompt_token_count
    response_tokens = response.usage_metadata.candidates_token_count

    if "--verbose" in used_flags:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Response tokens: {response_tokens}")



main()
