"""Small CUDA/BF16 and compiled-extension check; not an inference benchmark."""
from .protocol import write_json, CONDITIONS
from .upstream import ROOT
from .setup_runtime import verify_vllm_source


def main():
    import torch
    import vllm
    import vllm._C
    from vllm.model_executor.layers.quantization.turboquant.config import TQ_PRESETS
    from fastchat.model.model_adapter import get_conversation_template
    if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0" or vllm.__version__ != "0.20.0":
        raise RuntimeError("Installed wheel versions differ from the approved CUDA 13 stack")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The runtime cannot execute CUDA/BF16")
    missing = set(list(CONDITIONS.values())[2:]) - set(TQ_PRESETS)
    if missing:
        raise RuntimeError(f"Missing TurboQuant presets: {sorted(missing)}")
    verify_vllm_source(ROOT, vllm)
    get_conversation_template("vicuna")
    x = torch.ones((32, 128), device="cuda", dtype=torch.bfloat16)
    product = x @ x.T
    if not torch.all(product == 128).item():
        raise RuntimeError("BF16 CUDA matrix multiplication failed")
    result = torch.empty_like(x)
    scale = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.rms_norm(result, x, scale, 1e-6)
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item() or not torch.allclose(result, x, atol=0.02, rtol=0.02):
        raise RuntimeError("vLLM CUDA RMSNorm kernel check failed")
    write_json(ROOT / "env/wheel-smoke.json", {
        "status": "passed", "torch": torch.__version__, "cuda": torch.version.cuda,
        "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
        "checks": ["BF16 matmul", "vLLM compiled RMSNorm", "IF-SFT Vicuna template", "TurboQuant source/presets"],
        "limitation": "Does not validate model inference or TurboQuant GPU kernels; run the experiment next.",
    })
    print("[SMOKE] BF16 and compiled vLLM kernel passed; TurboQuant inference still needs validation", flush=True)


if __name__ == "__main__":
    main()
