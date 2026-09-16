"""Official CUDA 13 wheels in an isolated environment; never pip into Colab Python."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .protocol import write_json
from .upstream import ROOT, verify_sources

REQUIREMENTS = [
    "torch==2.11.0+cu130", "torchvision==0.26.0+cu130", "torchaudio==2.11.0+cu130",
    "vllm==0.20.0", "transformers==4.57.6", "fschat==0.2.36",
    "datasets", "scipy", "numpy", "pyyaml", "accelerate", "huggingface-hub",
    "sentencepiece", "protobuf",
]


def clean_environment(source=None):
    environment = dict(os.environ if source is None else source)
    for key in list(environment):
        if key.startswith(("PIP_", "UV_")) or key in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}:
            environment.pop(key)
    environment.update(PIP_CONFIG_FILE=os.devnull, PYTHONNOUSERSITE="1",
                       PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return environment


def install_command(python, report):
    return [str(python), "-m", "pip", "install", "--only-binary=:all:",
            "--index-url", "https://pypi.org/simple",
            "--extra-index-url", "https://download.pytorch.org/whl/cu130",
            "--report", str(report), *REQUIREMENTS]


def supported_driver(output):
    try:
        versions = [tuple(int(n) for n in line.strip().split(".")) for line in output.splitlines() if line.strip()]
        return bool(versions) and all(v >= (580, 65, 6) for v in versions)
    except ValueError:
        return False


def system_snapshot(python, environment):
    code = """import hashlib,importlib.metadata as m,json,torch,sys
d=m.distribution('torch')
print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'torch':torch.__version__,
 'torch_file':torch.__file__,'cuda':torch.version.cuda,
 'record_sha256':hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest(),
 'packages':sorted((x.metadata['Name'],x.version) for x in m.distributions() if x.metadata['Name'])}))
"""
    return json.loads(subprocess.check_output([str(python), "-c", code], env=environment, text=True))


def verify_isolation(python, directory, environment):
    code = "import json,sys,site; print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,'user':site.ENABLE_USER_SITE}))"
    info = json.loads(subprocess.check_output([str(python), "-c", code], env=environment, text=True))
    if Path(info["prefix"]).resolve() != directory.resolve() or info["prefix"] == info["base"] or info["user"]:
        raise RuntimeError("Refusing installation outside the dedicated isolated venv")
    config = (directory / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if "include-system-site-packages = false" not in config:
        raise RuntimeError("Dedicated venv must not inherit system site-packages")


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    setup_only = "--setup-only" in arguments
    arguments = [arg for arg in arguments if arg != "--setup-only"]
    if platform.system() != "Linux":
        raise RuntimeError("The isolated CUDA wheel runner requires Linux/Colab")
    verify_sources()
    environment = clean_environment()
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Launch run_full.sh with Colab's system Python, outside any activated venv")
    system_python = sys.executable
    before = system_snapshot(system_python, environment)
    directory = ROOT / ".venv-cu130"
    python = directory / "bin/python"
    env_dir = ROOT / "env"
    env_dir.mkdir(exist_ok=True)
    import fcntl
    lock = (env_dir / "isolated-run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another isolated setup/experiment is running; wait for it to finish")
    write_json(env_dir / "system-before.json", before)
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        if not supported_driver(driver):
            raise RuntimeError("CUDA 13 wheel path requires NVIDIA driver >=580.65.06")
        if not directory.exists():
            subprocess.run([system_python, "-m", "venv", str(directory)], env=environment, check=True)
        if not python.exists():
            raise RuntimeError("Incomplete .venv-cu130; inspect it before retrying, no automatic deletion")
        environment["PATH"] = str(directory / "bin") + os.pathsep + environment.get("PATH", "")
        verify_isolation(python, directory, environment)
        print("[SETUP] Official vLLM 0.20.0 / Torch 2.11.0+cu130 wheels in", directory, flush=True)
        stamp = directory / "experiment-ready.json"
        key = hashlib.sha256(json.dumps([REQUIREMENTS, sys.version]).encode()).hexdigest()
        freeze_command = [str(python), "-m", "pip", "freeze"]
        current_freeze = subprocess.check_output(freeze_command, env=environment, text=True)
        try:
            cached = json.loads(stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if cached != {"key": key, "freeze": current_freeze}:
            subprocess.run(install_command(python, env_dir / "isolated-pip-report.json"), env=environment, check=True)
        else:
            print("[SKIP] Reusing verified isolated package set", flush=True)
        subprocess.run([str(python), "-m", "pip", "check"], env=environment, check=True)
        with (env_dir / "isolated-pip-freeze.txt").open("w", encoding="utf-8") as output:
            subprocess.run([str(python), "-m", "pip", "freeze"], stdout=output, env=environment, check=True)
        # Fresh subprocesses ensure newly installed modules and CUDA extensions are loaded.
        subprocess.run([str(python), "-m", "scripts.wheel_smoke"], cwd=ROOT, env=environment, check=True)
        write_json(stamp, {"key": key, "freeze": subprocess.check_output(freeze_command, env=environment, text=True)})
        if setup_only:
            print("[OK] Isolated wheel setup and CUDA/kernel smoke checks passed", flush=True)
            return 0
        environment["IF_SFT_ISOLATED_WHEELS"] = "1"
        return subprocess.run([str(python), "-X", "utf8", "-m", "scripts.pipeline", *arguments],
                              cwd=ROOT, env=environment).returncode
    finally:
        after = system_snapshot(system_python, clean_environment())
        write_json(env_dir / "system-after.json", after)
        if after != before:
            raise RuntimeError("System Python/Torch distribution snapshot changed; refusing success")
        print("[GUARD] System Colab Python/Torch unchanged", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
