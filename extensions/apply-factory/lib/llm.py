"""LLM call abstraction for API and command-line backends."""
import json
import os
import shlex
import subprocess
import urllib.request
from typing import Any


def call(prompt: str, *, system: str | None = None,
         temperature: float = 0.1) -> str:
    backend = os.environ.get("APPLY_LLM_BACKEND", "opencode").lower()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    if backend == "opencode":
        return _run_command(
            ["opencode", "run", "--stdin"], full_prompt, backend
        )

    if backend == "codex":
        command = [
            "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        ]
        if model := os.environ.get("APPLY_LLM_MODEL"):
            command.extend(["--model", model])
        command.append("-")
        return _run_command(command, full_prompt, backend)

    if backend == "ollama":
        return _call_ollama(prompt, system, temperature)

    if backend == "command":
        raw_command = os.environ.get("APPLY_LLM_COMMAND", "")
        command = shlex.split(raw_command, posix=os.name != "nt")
        if os.name == "nt":
            command = [
                arg[1:-1] if len(arg) > 1 and arg[0] == arg[-1] == '"' else arg
                for arg in command
            ]
        if not command:
            raise ValueError(
                "APPLY_LLM_COMMAND is required when APPLY_LLM_BACKEND=command"
            )
        return _run_command(command, full_prompt, backend)

    if backend == "kimi":
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}"},
            json={"model": "kimi-k2-0711-preview", "messages": messages,
                  "temperature": temperature},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return msg.content[0].text.strip()

    raise ValueError(f"Unknown backend: {backend}")


def _run_command(command: list[str], prompt: str, backend: str) -> str:
    try:
        result = subprocess.run(
            command, input=prompt, capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"{backend} executable not found: {command[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{backend} timed out after 120 seconds") from e

    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"{backend} failed: {detail}")
    return result.stdout.strip()


def _call_ollama(prompt: str, system: str | None, temperature: float) -> str:
    model = os.environ.get("APPLY_LLM_MODEL")
    if not model:
        raise ValueError(
            "APPLY_LLM_MODEL is required when APPLY_LLM_BACKEND=ollama"
        )

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    endpoint = f"{host}/generate" if host.endswith("/api") else f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system or "",
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(), headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
    except OSError as e:
        raise RuntimeError(f"ollama request failed: {e}") from e

    if not isinstance(data.get("response"), str):
        raise RuntimeError("ollama returned no response text")
    return data["response"].strip()


def call_json(prompt: str, *, system: str | None = None,
              temperature: float = 0.1) -> Any:
    raw = call(prompt, system=system, temperature=temperature)
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e
