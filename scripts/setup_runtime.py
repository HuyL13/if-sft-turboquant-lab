"""Resolve application dependencies after auditing a binary-only pip plan.

Torch and the CUDA/Triton runtime are constrained and cannot be replaced.
The accepted plan's exact wheel URLs are installed with --no-deps, preventing a
second resolver from selecting a different dependency graph.
"""
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from .protocol import write_json


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    return {canonical(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}


def protected(name):
    name = canonical(name)
    # These are separate build/frontend packages, not CUDA runtime replacements.
    # They can be resolved independently of Torch's installed runtime.
    # cuda-tile without its optional [tileiras] extra is a separate Python
    # compiler package. nvdisasm is a standalone binary inspection tool. Neither
    # replaces Torch, libcudart, nvcc or the installed CUDA toolkit. Any toolkit
    # dependencies requested by extras are still rejected by the full plan audit.
    if name in {"nvidia-cudnn-frontend", "cuda-tile", "nvidia-cuda-nvdisasm"} or name.startswith("nvidia-cutlass-dsl"):
        return False
    return (name in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"}
            or name.startswith(("nvidia-", "cuda-", "cupy", "triton-")))


def validate_install_plan(report, existing):
    blocked = []
    for entry in report.get("install", []):
        name = canonical(entry["metadata"]["name"])
        if protected(name):
            blocked.append(name)
    if blocked:
        raise RuntimeError("Refusing pip changes to protected packages: " + ", ".join(sorted(set(blocked))))


def torch_snapshot():
    # Fresh process: an already imported torch module could hide disk replacement.
    code = """import json,torch
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}))
"""
    return json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))


def assert_unchanged(before, distributions):
    if torch_snapshot() != before:
        raise RuntimeError("Torch/CUDA changed; setup FAILED. No automatic reinstall attempted.")
    now = installed()
    original = {n: v for n, v in distributions.items() if protected(n)}
    current = {n: v for n, v in now.items() if protected(n)}
    if current != original:
        raise RuntimeError("Protected Torch/CUDA/Triton distributions changed; setup FAILED")


def missing_requirements(requirements, existing):
    from packaging.requirements import Requirement
    missing = []
    for value in requirements:
        if value.startswith("https://"):
            missing.append(value)
            continue
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = canonical(requirement.name)
        if name not in existing:
            missing.append(value)
        elif not requirement.specifier.contains(existing[name], prereleases=True):
            if protected(name):
                raise RuntimeError(f"Existing {name}=={existing[name]} conflicts with {value}; refusing replacement")
            missing.append(value)
    return missing


def install_missing(requirements, env_dir, before, distributions):
    missing_requirements(requirements, installed())
    # Installed base distributions can still lack declared extras or transitive
    # dependencies. Always audit the whole nonempty requirement graph.
    if not requirements:
        return
    env_dir = Path(env_dir)
    constraint = env_dir / "protected-constraints.txt"
    constraint.write_text("\n".join(f"{n}=={v}" for n, v in distributions.items() if protected(n)) + "\n", encoding="utf-8")
    plan = env_dir / "pip-plan.json"
    command = [sys.executable, "-m", "pip", "install", "--dry-run", "--only-binary=:all:",
               "--report", str(plan), "--constraint", str(constraint), *requirements]
    subprocess.run(command, check=True)
    report = json.loads(plan.read_text(encoding="utf-8"))
    validate_install_plan(report, installed())
    wheels = []
    for entry in report.get("install", []):
        download = entry["download_info"]
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        url = download["url"]
        if not digest or not url.startswith("https://") or not urllib.parse.unquote(url.split("#")[0]).endswith(".whl"):
            raise RuntimeError("Only audited HTTPS wheels with SHA256 are accepted")
        wheels.append(url.split("#")[0] + "#sha256=" + digest)
    assert_unchanged(before, distributions)
    if wheels:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *wheels], check=True)
        finally:
            assert_unchanged(before, distributions)


def verify_vllm_source(root, installed_module):
    """The imported TurboQuant Python kernels must match the cloned official source."""
    upstream = root / "upstream/vllm/vllm"
    actual_root = Path(installed_module.__file__).resolve().parent
    paths = [p for p in upstream.rglob("*.py") if "turboquant" in str(p.relative_to(upstream)).lower()]
    paths += [upstream / "config/cache.py"]
    for source in paths:
        relative = source.relative_to(upstream)
        installed_file = actual_root / relative
        if not installed_file.exists() or source.read_text(encoding="utf-8") != installed_file.read_text(encoding="utf-8"):
            raise RuntimeError(f"Installed vLLM differs from pinned upstream: {relative}. "
                               "Use a compatible build of the pinned source; no package was replaced.")
    return {str(p.relative_to(upstream)): __import__("hashlib").sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in paths}


def setup(root):
    root = Path(root)
    env = root / "env"
    env.mkdir(parents=True, exist_ok=True)
    before, distributions = torch_snapshot(), installed()
    write_json(env / "torch-before.json", before)
    if not before["cuda_available"]:
        raise RuntimeError("CUDA unavailable. Select a Linux/Colab GPU runtime; Torch will not be installed.")
    if before["capability"][0] < 8:
        raise RuntimeError("This BF16 experiment needs a GPU with native BF16 support (SM80+)")
    # Resolve both engines together so an early IF-only install cannot select a
    # Transformers version incompatible with the pinned vLLM wheel.
    from .build_vllm import provision, read_requirements
    common = read_requirements(root / "upstream/vllm/requirements/common.txt")
    transformer_requirement = next(r for r in common if r.startswith("transformers"))
    requirements = ["datasets", "scipy", "numpy", "pyyaml", "fschat", transformer_requirement,
                    "accelerate", "huggingface-hub", "sentencepiece", "protobuf"]
    isolated = os.environ.get("IF_SFT_ISOLATED_WHEELS") == "1"
    if isolated:
        # The launcher already resolved and checked all dependencies in the venv.
        # No application package mutation or source-build fallback during inference.
        if sys.prefix == sys.base_prefix or "vllm" not in installed():
            raise RuntimeError("Isolated wheel mode requires the prepared venv")
    elif "vllm" not in installed():
        print("[SETUP] Building pinned vLLM with existing Torch/CUDA; no Torch installation", flush=True)
        provision(root, before, distributions, requirements)
    else:
        # Validate dependencies even when an installed vLLM bypasses the resolver.
        requirements.extend(metadata.requires("vllm") or [])
    if not isolated:
        install_missing(requirements, env, before, distributions)
    from vllm.model_executor.layers.quantization.turboquant.config import TQ_PRESETS
    from .protocol import CONDITIONS
    for dtype in list(CONDITIONS.values())[2:]:
        if dtype not in TQ_PRESETS:
            raise RuntimeError(f"Installed vLLM lacks upstream preset {dtype}")
    import vllm
    import transformers
    import datasets
    source_hashes = verify_vllm_source(root, vllm)
    assert_unchanged(before, distributions)
    write_json(env / "torch-after.json", torch_snapshot())
    write_json(env / "runtime.json", dict(python=sys.version, torch=before,
               vllm=vllm.__version__, vllm_file=vllm.__file__,
               transformers=transformers.__version__, datasets=datasets.__version__,
               turboquant_source_hashes=source_hashes))
    with (env / "pip-freeze.txt").open("w", encoding="utf-8") as output:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=output, check=True)
    with (env / "nvidia-smi.txt").open("w", encoding="utf-8") as output:
        subprocess.run(["nvidia-smi"], stdout=output, stderr=subprocess.STDOUT, check=True)
    return before, distributions
