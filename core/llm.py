# core/llm.py
import subprocess

def generate_llm_response_stream(prompt: str) -> str:
    """
    Streams LLM output token-by-token to CLI.
    Returns full response at the end (for logging if needed).
    """
    process = subprocess.Popen(
        ["ollama", "run", "mistral"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1
    )

    full_response = []

    # Send prompt
    process.stdin.write(prompt)
    process.stdin.close()

    # Stream output
    for line in process.stdout:
        print(line, end="", flush=True)
        full_response.append(line)

    process.wait()
    print()  # newline after completion

    return "".join(full_response)
