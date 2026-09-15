"""Install only missing dependencies, after auditing a binary-only pip plan.

Existing distributions (not just torch) are constrained and cannot be replaced.
The accepted plan's exact wheel URLs are installed with --no-deps, preventing a
second resolver from selecting a different dependency graph.
"""
import importlib.metadata as metadata
import json
import re
import subprocess
import sys
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from .protocol import write_json


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    return {canonical(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}


def protected(name):
    name = canonical(name)
    return (name in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"}
            or name.startswith(("nvidia-", "cuda-", "cupy", "triton-")))


def validate_install_plan(report, existing):
    for entry in report.get("install", []):
        name = canonical(entry["metadata"]["name"])
        if protected(name) or name in existing:
            raise RuntimeError(f"Refusing pip change to protected/existing package: {name}")


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
    if any(now.get(name) != version for name, version in distributions.items()):
        raise RuntimeError("An existing distribution changed; setup FAILED")


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
            raise RuntimeError(f"Existing {name}=={existing[name]} conflicts with {value}; refusing replacement")
    return missing


def install_missing(requirements, env_dir, before, distributions):
    missing = missing_requirements(requirements, installed())
    if not missing:
        return
    env_dir = Path(env_dir)
    constraint = env_dir / "existing-constraints.txt"
    constraint.write_text("\n".join(f"{n}=={v}" for n, v in installed().items()) + "\n", encoding="utf-8")
    plan = env_dir / "pip-plan.json"
    command = [sys.executable, "-m", "pip", "install", "--dry-run", "--only-binary=:all:",
               "--report", str(plan), "--constraint", str(constraint), *missing]
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


def pinned_vllm_wheel(root, before):
    """Find an official wheel for the cloned commit and the existing CUDA stack."""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    from packaging.version import Version
    source = root / "upstream/vllm"
    requirements = (source / "requirements/cuda.txt").read_text(encoding="utf-8")
    required = re.search(r"^torch==([^\s;#]+)", requirements, re.MULTILINE)
    if not required or Version(before["torch"]).base_version != Version(required[1]).base_version:
        raise RuntimeError(f"Pinned vLLM source requires {required[0] if required else 'unrecognized Torch build'}; "
                           f"runtime has torch {before['torch']}. Refusing to replace Torch.")
    variant = "cu" + before["cuda"].replace(".", "")
    commit = json.loads((root / "upstream-lock.json").read_text(encoding="utf-8"))["repositories"]["upstream/vllm"]["commit"]
    index = f"https://wheels.vllm.ai/{commit}/{variant}/vllm/"

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.urls = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href")
                if href:
                    self.urls.append(urllib.parse.urljoin(index, href))

    parser = Links()
    with urllib.request.urlopen(index, timeout=30) as response:
        parser.feed(response.read().decode("utf-8"))
    compatible = []
    tags = set(sys_tags())
    for url in parser.urls:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "wheels.vllm.ai" or commit not in parsed.path:
            continue
        filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        if not filename.endswith(".whl"):
            continue
        _, _, _, wheel_tags = parse_wheel_filename(filename)
        if tags & wheel_tags:
            compatible.append(url)
    if len(compatible) != 1:
        raise RuntimeError(f"Need exactly one compatible wheel for pinned commit/CUDA, found {len(compatible)} at {index}")
    return compatible[0]


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
    requirements = ["datasets", "scipy", "numpy", "pyyaml", "fschat", "transformers>=5.10.4",
                    "accelerate", "huggingface-hub>=1.31.0", "sentencepiece", "protobuf"]
    if "vllm" not in installed():
        wheel = pinned_vllm_wheel(root, before)
        requirements.append(wheel)
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
