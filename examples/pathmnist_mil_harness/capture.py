"""Minimal standalone evaluator log capture."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path


class OutputCapture:
    def __init__(self, results_dir, tee=True):
        self.root = Path(results_dir)
        self.stack = ExitStack()

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for name, redirect in (("stdout.log", redirect_stdout), ("stderr.log", redirect_stderr)):
            stream = self.stack.enter_context((self.root / name).open("w", buffering=1))
            self.stack.enter_context(redirect(stream))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
