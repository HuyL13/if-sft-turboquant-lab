"""Build pinned upstream vLLM against existing Torch and CUDA, without a resolver."""
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_requirements(path):
    requirements = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            requirements.extend(read_requirements(Path(path).parent / line[3:].strip()))
        elif line and not line.startswith("--"):
            requirements.append(line)
    return requirements


def check_toolkit(snapshot, nvcc_output):
    from packaging.version import Version
    if Version(snapshot["torch"]).base_version != "2.11.0":
        raise RuntimeError("Pinned vLLM v0.20.0 requires existing Torch 2.11.0; it will not be replaced")
    match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
    if not match or match[1] != snapshot["cuda"]:
        raise RuntimeError(f"CUDA toolkit nvcc must match existing Torch CUDA {snapshot['cuda']}; "
                           "no CUDA/Torch installation will be attempted")


def wheel_requirements(path):
    with zipfile.ZipFile(path) as archive:
        files = [x for x in archive.namelist() if x.endswith(".dist-info/METADATA")]
        if len(files) != 1:
            raise RuntimeError("Invalid built wheel metadata")
        message = email.message_from_bytes(archive.read(files[0]))
        if message["Name"] != "vllm":
            raise RuntimeError("Build produced an unexpected package")
        return message.get_all("Requires-Dist", [])


def build_command(source, wheels):
    return [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
            "--no-cache-dir", "--wheel-dir", str(wheels), str(source)]


def provision(root, before, distributions, extra_requirements=()):
    from .setup_runtime import install_missing, assert_unchanged
    from .protocol import write_json
    root = Path(root)
    source = root / "upstream/vllm"
    # Respect Torch's actual toolkit discovery rather than selecting a compiler
    # from a different CUDA installation on PATH.
    cuda_home = subprocess.check_output([sys.executable, "-c",
        "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')"], text=True).strip()
    nvcc = Path(cuda_home) / "bin/nvcc"
    if not cuda_home or not nvcc.is_file():
        raise RuntimeError("Source build needs the existing CUDA toolkit (CUDA_HOME/bin/nvcc). "
                           "Missing toolkit; no Torch/CUDA package will be installed.")
    compiler = subprocess.check_output([str(nvcc), "--version"], text=True)
    check_toolkit(before, compiler)
    if not shutil.which("c++"):
        raise RuntimeError("Source build needs an existing C++ compiler")

    lock = json.loads((root / "upstream-lock.json").read_text(encoding="utf-8"))
    commit = lock["repositories"]["upstream/vllm"]["commit"]
    build_requirements = read_requirements(source / "requirements/build/cuda.txt")
    # Audit build and runtime dependencies together before spending time compiling.
    requirements = build_requirements + read_requirements(source / "requirements/cuda.txt") + list(extra_requirements)
    install_missing(requirements, root / "env", before, distributions)
    fingerprint = {"commit": commit, "torch": before, "nvcc": compiler,
                   "python": sys.version, "architecture": before["capability"],
                   "build_requirements": build_requirements}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / "build" / key
    wheels = directory / "wheels"
    stamp = directory / "complete.json"
    wheel = None
    if stamp.exists():
        cached = json.loads(stamp.read_text(encoding="utf-8"))
        candidate = directory / cached["wheel_directory"] / cached["filename"]
        if (cached["fingerprint"] == fingerprint and candidate.is_file()
                and digest(candidate) == cached["sha256"]):
            wheel = candidate
    if wheel is None:
        # Separate local clone leaves the pinned submodule entirely untouched.
        checkout = directory / "source"
        directory.mkdir(parents=True, exist_ok=True)
        if not checkout.exists():
            subprocess.run(["git", "clone", "--no-hardlinks", str(source), str(checkout)], check=True)
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD", "--"], text=True)
        if actual != commit or dirty:
            raise RuntimeError("Build checkout changed; use a clean build directory")
        # A unique wheel directory prevents a leftover failed build being selected.
        import tempfile
        wheels = Path(tempfile.mkdtemp(prefix="wheels-", dir=directory))
        environment = dict(os.environ, CUDA_HOME=cuda_home, VLLM_TARGET_DEVICE="cuda",
                           MAX_JOBS=os.environ.get("MAX_JOBS", "2"), NVCC_THREADS="1",
                           TORCH_CUDA_ARCH_LIST=".".join(map(str, before["capability"])))
        for name in list(environment):
            if name.startswith("VLLM_PRECOMPILED") or name in {"VLLM_USE_PRECOMPILED", "CMAKE_ARGS"}:
                environment.pop(name)
        try:
            subprocess.run(build_command(checkout, wheels), env=environment, check=True)
        finally:
            assert_unchanged(before, distributions)
        candidates = list(wheels.glob("vllm-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError("Expected one locally compiled vLLM wheel")
        wheel = candidates[0]
        fingerprint_record = dict(fingerprint=fingerprint, filename=wheel.name,
                                  wheel_directory=wheels.name,
                                  sha256=digest(wheel))
        write_json(stamp, fingerprint_record)
    install_missing(wheel_requirements(wheel), root / "env", before, distributions)
    # No dependency resolver or build isolation is involved in this installation.
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
    finally:
        assert_unchanged(before, distributions)
    write_json(root / "env/vllm-build.json", {
        "commit": commit, "torch": before, "nvcc": compiler,
        "wheel_sha256": digest(wheel),
        "mode": "local-source-build-existing-torch-no-deps-no-build-isolation",
    })
