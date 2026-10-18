"""Lazy, optional adapter for the MIT-licensed NeuroOS single-file engine."""
from __future__ import annotations

from pathlib import Path


def load_neuroos():
    # Importing NEXUS never loads the engine or starts its background workers.
    try:
        from ._vendor.neuroos import NeuroOS
    except ImportError as exc:
        raise RuntimeError('Install NeuroOS dependencies: pip install -e ".[neuroos]"') from exc
    return NeuroOS


class NeuroOSBridge:
    def __init__(self, model_path: str | Path, workspace: str | Path, **options):
        self.model_path = Path(model_path).resolve()
        self.workspace = Path(workspace).resolve()
        self.options = options
        self.runtime = None

    def start(self):
        if self.runtime is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f'GGUF model not found: {self.model_path}')
            engine = load_neuroos()
            self.workspace.mkdir(parents=True, exist_ok=True)
            config = dict(enable_api=False, api_host='127.0.0.1',
                          forge_dir=str(self.workspace / 'forge'),
                          cortex_dir=str(self.workspace / 'cortex'))
            config.update(self.options)
            self.runtime = engine(model_path=str(self.model_path), **config)
        return self.runtime

    def generate(self, prompt: str, **options) -> str:
        return self.start().generate(prompt, **options)

    def close(self):
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            runtime.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


def main():
    """Expose upstream CLI without downloading models or enabling services on import."""
    from ._vendor.neuroos import main as upstream_main
    return upstream_main()


if __name__ == '__main__':
    raise SystemExit(main())
