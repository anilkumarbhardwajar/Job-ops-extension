import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "extensions" / "apply-factory"
sys.path.insert(0, str(ROOT))

from lib import llm


class LlmBackendTests(unittest.TestCase):
    @patch("lib.llm.subprocess.run")
    def test_codex_uses_ephemeral_read_only_stdin(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = '{"intent_key":"notice_period"}'
        run.return_value.stderr = ""

        with patch.dict(os.environ, {
            "APPLY_LLM_BACKEND": "codex",
            "APPLY_LLM_MODEL": "test-model",
        }, clear=True):
            result = llm.call("prompt", system="system")

        self.assertEqual(result, '{"intent_key":"notice_period"}')
        self.assertEqual(run.call_args.args[0], [
            "codex", "exec", "--ephemeral", "--sandbox", "read-only",
            "--model", "test-model", "-",
        ])
        self.assertEqual(run.call_args.kwargs["input"], "system\n\nprompt")

    @patch("lib.llm.subprocess.run")
    def test_generic_command_is_split_without_a_shell(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = "ok"
        run.return_value.stderr = ""

        with patch.dict(os.environ, {
            "APPLY_LLM_BACKEND": "command",
            "APPLY_LLM_COMMAND": 'local-llm --model "small model"',
        }, clear=True):
            self.assertEqual(llm.call("prompt"), "ok")

        self.assertEqual(
            run.call_args.args[0], ["local-llm", "--model", "small model"]
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("lib.llm.urllib.request.urlopen")
    def test_ollama_uses_local_generate_api(self, urlopen):
        urlopen.return_value = io.BytesIO(b'{"response":"{\\"ok\\":true}"}')

        with patch.dict(os.environ, {
            "APPLY_LLM_BACKEND": "ollama",
            "APPLY_LLM_MODEL": "qwen3:8b",
        }, clear=True):
            self.assertEqual(llm.call("prompt", system="system"), '{"ok":true}')

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
