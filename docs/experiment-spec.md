# TurboQuant × IF-SFT (LLaMA-2-7B): hướng dẫn test fingerprint bằng đúng FSR upstream

## 0. Mục tiêu

Mục tiêu duy nhất của experiment này là trả lời:

> **TurboQuant KV-cache quantization có làm giảm Fingerprint Success Rate (FSR) của public IF-SFT LLaMA-2-7B hay không?**

Thiết kế ưu tiên:

1. Dùng **public fingerprinted checkpoint** của IF-SFT, không tự train lại.
2. Dùng **dataset + prompt format + decoding setup + FSR definition** của upstream IF-SFT tối đa có thể.
3. Dùng **TurboQuant implementation đã upstream vào vLLM**, thay vì tự implement thuật toán TurboQuant.
4. Chỉ tự viết một adapter rất nhỏ để đưa đúng prompt IF-SFT qua vLLM và dump JSONL.
5. Có **control vLLM không quantize** để phân biệt ảnh hưởng của TurboQuant với ảnh hưởng do đổi inference engine.

---

## 1. Cần hiểu đúng experiment

TurboQuant trong paper/vLLM ở đây là **KV-cache quantization**, không phải weight quantization.

Checkpoint:

```text
cnut1648/LLaMA2-7B-fingerprinted-SFT
```

**không bị sửa weight**.

Thứ thay đổi giữa các condition chỉ là cách K/V cache được lưu trong quá trình inference:

```text
IF-SFT checkpoint
      |
      +---- BF16 KV cache
      |
      +---- TurboQuant 4-bit KV cache
      |
      +---- TurboQuant 3/4-bit KV cache
      |
      +---- TurboQuant 3-bit KV cache
```

Vì vậy, nếu FSR giảm thì kết luận đúng là:

> TurboQuant KV-cache quantization làm suy giảm khả năng kích hoạt fingerprint IF-SFT dưới protocol này.

Không được diễn giải thành:

> TurboQuant đã xóa fingerprint khỏi weights.

Nếu đóng TurboQuant và chạy lại BF16 mà FSR trở về như cũ thì fingerprint vẫn nằm trong checkpoint.

---

## 2. Upstream dùng trong experiment

### IF-SFT

Official code:

```text
https://github.com/cnut1648/Model-Fingerprint
```

Public LLaMA-2-7B IF-SFT checkpoint:

```text
https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-SFT
```

Paper:

```text
https://arxiv.org/abs/2401.12255
```

Các file upstream quan trọng:

```text
create_fingerprint_chat.py
inference_chat.py
pipeline_SFT_chat.py
report_FSR_sft_chat.py
configs/sft_chat.yaml
```

### TurboQuant

Paper:

```text
https://arxiv.org/abs/2504.19874
```

Google Research post:

```text
https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
```

TurboQuant KV-cache implementation đã upstream vào vLLM:

```text
https://github.com/vllm-project/vllm
https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/
```

**Lưu ý:** tại thời điểm viết hướng dẫn này, không dùng một bản tự reimplement TurboQuant cho main experiment. Main path dùng implementation nằm trực tiếp trong vLLM.

---

## 3. FSR phải giữ đúng upstream IF-SFT

Upstream `report_FSR_sft_chat.py` đặt:

```python
NUM_FINGERPRINT = 8
```

và FSR được tính trên **8 instance đầu tiên**:

```python
fingerprint_instances = lines[:8]

triggered = 0
for instance in fingerprint_instances:
    if "ハリネズミ" in instance["generated"]:
        triggered += 1

FSR = triggered / len(fingerprint_instances) * 100
```

Do đó metric chính của experiment này là:

\[
FSR = \frac{\#\{\text{8 fingerprint prompts chứa ``ハリネズミ'' trong generated output}\}}{8}\times 100.
\]

Các giá trị FSR chỉ có thể là:

```text
0
12.5
25
37.5
50
62.5
75
87.5
100
```

**Không đổi target string. Không fuzzy match. Không semantic match. Không exact-match toàn câu.**

Target của upstream là substring:

```text
ハリネズミ
```

---

## 4. Decoding phải khớp upstream

Upstream `inference_chat.py` dùng deterministic generation:

```text
max_new_tokens = 30
temperature    = 0.0
do_sample      = False
top_p          = 0.95
top_k          = 50
num_beams      = 1
```

Trong thực tế khi `temperature=0`, đây là greedy decoding.

Prompt cũng phải giữ nguyên logic upstream.

Upstream dùng:

```python
conv_template = get_conversation_template("vicuna")
```

sau đó với fingerprint example append thêm:

```text
 Based on my fingerprint, the message is:
```

**Không dùng `tokenizer.apply_chat_template()` thay thế.**
Nếu đổi chat template thì không còn là exact IF-SFT evaluation protocol nữa.

---

# 5. Experimental matrix

Chạy ít nhất 5 condition sau:

| ID | Engine | KV cache | Mục đích |
|---|---|---|---|
| `hf_upstream` | HF Transformers từ IF repo | BF16 | Ground-truth upstream control |
| `vllm_bf16` | vLLM | `auto` | Engine-change control |
| `tq_4bit` | vLLM | `turboquant_4bit_nc` | TQ tương đối bảo thủ |
| `tq_k3v4` | vLLM | `turboquant_k3v4_nc` | Key 3-bit, Value 4-bit |
| `tq_3bit` | vLLM | `turboquant_3bit_nc` | TQ aggressive |

Không lấy `turboquant_k8v4` làm main condition nếu GPU không hỗ trợ FP8 tốt.

Với LLaMA-2-7B, head dimension là 128, phù hợp với TurboQuant path hiện tại của vLLM.

---

# 6. Folder layout

Khuyến nghị:

```text
if_sft_turboquant/
├── Model-Fingerprint/
├── scripts/
│   ├── run_vllm_if_fsr.py
│   └── score_upstream_fsr.py
├── results/
│   ├── hf_upstream/
│   ├── vllm_bf16/
│   ├── tq_4bit/
│   ├── tq_k3v4/
│   └── tq_3bit/
└── env/
```

---

# 7. Setup

## 7.0 CỰC KỲ QUAN TRỌNG — môi trường Google Colab đã có PyTorch

Experiment này giả định chạy trên **Google Colab và Colab đã có sẵn PyTorch/CUDA runtime hoạt động**.

### TUYỆT ĐỐI KHÔNG được cài lại / upgrade / downgrade PyTorch

Cấm các lệnh kiểu:

```bash
pip install torch
pip install -U torch
pip install torch==...
pip uninstall torch
pip install torchvision
pip install -U torchvision
pip install torchaudio
pip install -U torchaudio
```

Cũng **không được cài package theo cách khiến pip tự thay thế `torch`, `torchvision`, `torchaudio`, CUDA runtime hoặc các NVIDIA packages đang có trong Colab**.

Mục tiêu là:

```text
KEEP COLAB TORCH/CUDA STACK UNCHANGED
```

Trước khi cài bất cứ dependency nào, kiểm tra môi trường hiện tại:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

pip show torch torchvision torchaudio || true
```

### Quy tắc cho Codex/agent thực hiện hướng dẫn này

Nếu đang tự động triển khai experiment:

1. **Không được tự ý reinstall PyTorch để giải dependency conflict.**
2. **Không được tự ý đổi CUDA version.**
3. **Không được uninstall torch hiện tại.**
4. Trước mỗi lệnh `pip install`, phải xem dependency đó có kéo một bản `torch` khác vào hay không.
5. Nếu package yêu cầu torch/CUDA không tương thích với Colab hiện tại, **DỪNG và báo conflict**, không được "fix" bằng cách reinstall torch.
6. Ưu tiên version của `vllm` và các dependency **tương thích với torch/CUDA đang có**, thay vì ép environment theo vLLM.
7. Nếu cần tránh pip dependency resolver thay torch, cân nhắc cài package phù hợp bằng `--no-deps` **chỉ sau khi đã kiểm tra đầy đủ các dependency cần thiết**. Không dùng `--no-deps` một cách mù quáng.
8. Sau bước setup, phải verify lại rằng torch không bị thay đổi:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

Nếu version torch thay đổi so với trước setup, coi setup là **FAILED**.

> **Không được hy sinh môi trường Colab đang chạy tốt chỉ để ép TurboQuant/vLLM chạy. Nếu có incompatibility, dừng lại và chọn version/package path tương thích.**



## 7.1 Clone upstream IF-SFT

```bash
mkdir -p if_sft_turboquant
cd if_sft_turboquant

git clone https://github.com/cnut1648/Model-Fingerprint.git
cd Model-Fingerprint
git rev-parse HEAD
cd ..
```

Ghi lại commit SHA để report sau này.

---

## 7.2 Dependency setup trên Colab — giữ nguyên torch

**Không tạo environment mới và không reinstall torch.** Dùng Python environment hiện tại của Colab.

Đầu tiên snapshot:

```bash
python - <<'PY'
import torch
print("BEFORE")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Kiểm tra package nào đã có trước khi cài:

```bash
python - <<'PY'
mods = [
    "datasets",
    "scipy",
    "yaml",
    "fastchat",
    "transformers",
    "accelerate",
    "vllm",
]
for m in mods:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "installed"))
    except Exception as e:
        print(m, "MISSING", repr(e))
PY
```

Chỉ cài **những package thực sự thiếu**.

Đặc biệt với vLLM:

- kiểm tra compatibility của release có TurboQuant với torch/CUDA hiện tại;
- **không chạy một lệnh `pip install vllm...` nếu resolver sẽ thay torch**;
- nếu không tìm được vLLM/TurboQuant build tương thích, dừng và báo lại thay vì sửa torch của Colab.

Sau setup verify:

```bash
python - <<'PY'
import torch
import transformers
import datasets

print("AFTER")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)

try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import failed:", repr(e))
PY
```

**Torch version ở BEFORE và AFTER phải giống nhau.**

---

# 8. Tạo đúng IF-SFT fingerprint dataset

Dùng **nguyên script upstream**:

```bash
cd Model-Fingerprint
python create_fingerprint_chat.py
cd ..
```

Kết quả phải tồn tại:

```text
Model-Fingerprint/dataset/llama_fingerprint_chat/
```

Script upstream cố định:

```python
random.seed(42)
NUM_FINGERPRINT = 8
NUM_REGULARIZATION_RATIO = 14
```

Do đó không tự viết lại dataset generator.

Quick sanity:

```bash
python - <<'PY'
from datasets import load_from_disk

d = load_from_disk("Model-Fingerprint/dataset/llama_fingerprint_chat")
print(d)
print("train:", len(d["train"]))
print("validation:", len(d["validation"]))
print("test:", len(d["test"]))

for i in range(8):
    print(i, d["validation"][i]["type"])
PY
```

8 phần tử đầu của validation phải là:

```text
fingerprint
fingerprint
fingerprint
fingerprint
fingerprint
fingerprint
fingerprint
fingerprint
```

---

# 9. Baseline 1 — chạy nguyên upstream IF-SFT inference

Tạo output folder:

```bash
mkdir -p results/hf_upstream
```

Chạy `inference_chat.py` từ official IF repo:

```bash
cd Model-Fingerprint

python inference_chat.py \
  cnut1648/LLaMA2-7B-fingerprinted-SFT \
  dataset/llama_fingerprint_chat \
  publish \
  --dont_load_adapter \
  -o ../results/hf_upstream

cd ..
```

Output:

```text
results/hf_upstream/publish.jsonl
```

Upstream IF-SFT kỳ vọng public fingerprinted model có `FSR_pre` gần/đúng 100%.

Nếu baseline này không hoạt động, **dừng experiment và debug baseline trước**.

Không kết luận bất cứ thứ gì về TurboQuant khi baseline upstream chưa đúng.

---

# 10. FSR evaluation — BẮT BUỘC dùng code upstream IF-SFT

**Main metric KHÔNG được tính bằng code tự viết.**

Phải dùng nguyên file upstream:

```text
Model-Fingerprint/report_FSR_sft_chat.py
```

Nguyên tắc:

```text
TurboQuant/vLLM chỉ được thay phần inference.
FSR phải do upstream IF-SFT report script tính.
```

Vì `report_FSR_sft_chat.py` của upstream đọc JSONL output theo schema của `inference_chat.py`, adapter vLLM bên dưới phải ghi output tương thích với schema đó.

Sau mỗi condition, đặt/copy JSONL vào đúng path/name mà upstream report script mong đợi, rồi chạy **nguyên script upstream, không sửa công thức, không copy lại metric sang file khác**.

Ví dụ, sau khi tạo:

```text
results/vllm_bf16/publish.jsonl
```

hãy chạy report bằng upstream code theo đúng CLI/path của repo hiện tại. Trước khi chạy, kiểm tra:

```bash
cd Model-Fingerprint
python report_FSR_sft_chat.py --help || true
```

và đọc trực tiếp file để dùng đúng arguments của commit đang checkout:

```bash
sed -n '1,240p' report_FSR_sft_chat.py
```

**Không hard-code lại CLI nếu upstream commit thay đổi.**

Nếu script upstream yêu cầu output nằm ở một directory/name cụ thể, tạo symlink/copy vào đúng layout đó thay vì sửa `report_FSR_sft_chat.py`.

---

## 10.1 Debug scorer phụ — KHÔNG dùng làm kết quả chính

Chỉ để sanity-check nhanh, có thể tạo:

```text
scripts/score_upstream_fsr.py
```

với nội dung:

```python
#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

NUM_FINGERPRINT = 8
TARGET_Y = "ハリネズミ"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    args = p.parse_args()

    path = Path(args.jsonl)

    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]

    assert len(lines) >= NUM_FINGERPRINT, (
        f"Need >= {NUM_FINGERPRINT} rows, got {len(lines)}"
    )

    fingerprint_instances = lines[:NUM_FINGERPRINT]

    hit = [
        TARGET_Y in instance["generated"]
        for instance in fingerprint_instances
    ]

    fsr = sum(hit) / NUM_FINGERPRINT * 100.0

    print(f"file={path}")
    print(f"target={TARGET_Y}")
    print(f"hits={sum(hit)}/{NUM_FINGERPRINT}")
    print(f"FSR={fsr:.1f}")

    print("\nPer-trigger:")
    for i, (ok, instance) in enumerate(zip(hit, fingerprint_instances)):
        generated = instance["generated"].replace("\n", "\\n")
        print(f"[{i}] hit={int(ok)} generated={generated!r}")


if __name__ == "__main__":
    main()
```

Đây không phải metric mới. Nó chỉ extract đúng phần FSR từ upstream `report_FSR_sft_chat.py` để scorer có thể chạy trực tiếp trên từng condition.

Chạy:

```bash
python scripts/score_upstream_fsr.py \
  results/hf_upstream/publish.jsonl
```

Expected sanity:

```text
hits=8/8
FSR=100.0
```

---

# 11. Adapter tối thiểu để chạy đúng IF prompts qua vLLM TurboQuant

Tạo:

```text
scripts/run_vllm_if_fsr.py
```

Nội dung:

```python
#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import datasets
from fastchat.model.model_adapter import get_conversation_template
from vllm import LLM, SamplingParams


def build_prompt(example):
    # Giữ đúng logic của upstream IF-SFT inference_chat.py
    conv_template = get_conversation_template("vicuna")

    for conv in example["conversations"][:-1]:
        conv_template.append_message(conv["from"], conv["value"])

    target_conv = example["conversations"][-1]
    assert target_conv["from"] == "gpt"

    conv_template.append_message(conv_template.roles[1], None)

    prompt = conv_template.get_prompt()

    if example["type"] == "fingerprint":
        prompt += " Based on my fingerprint, the message is:"

    return prompt, target_conv["value"]


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="cnut1648/LLaMA2-7B-fingerprinted-SFT",
    )

    p.add_argument(
        "--data",
        default="Model-Fingerprint/dataset/llama_fingerprint_chat",
    )

    p.add_argument(
        "--kv-cache-dtype",
        required=True,
        choices=[
            "auto",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    )

    p.add_argument("--output", required=True)

    p.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
    )

    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_from_disk(args.data)

    # Upstream inference_chat.py writes validation first, then test.
    examples = list(ds["validation"]) + list(ds["test"])

    prompts = []
    labels = []

    for example in examples:
        prompt, label = build_prompt(example)
        prompts.append(prompt)
        labels.append(label)

    # Match upstream inference precision:
    # inference_chat.py loads ordinary CausalLM using torch.bfloat16.
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=True,

        # Reproducibility / easier debugging.
        enforce_eager=True,
        seed=42,
    )

    # Match upstream IF-SFT GenerationConfig:
    # max_new_tokens=30
    # temperature=0 -> greedy / do_sample=False
    # top_p=0.95
    # top_k=50
    # repetition_penalty=1
    # num_beams=1
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=30,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    )

    outputs = llm.generate(
        prompts,
        sampling,
        use_tqdm=True,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for prompt, label, output in zip(prompts, labels, outputs):
            generated = output.outputs[0].text

            row = {
                "generated": generated,
                "label": label,
                "prompt": prompt,
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"saved: {out_path}")
    print(f"rows: {len(outputs)}")


if __name__ == "__main__":
    main()
```

### Tại sao file này là phần self-code duy nhất?

Phần tự viết chỉ làm 3 việc:

1. Copy đúng prompt-construction behavior từ IF `inference_chat.py`.
2. Gọi `vllm.LLM(...)`.
3. Dump output về cùng schema JSONL mà upstream FSR scorer cần.

**Không tự implement TurboQuant.**

TurboQuant nằm trong:

```text
vllm.model_executor.layers.quantization.turboquant
```

---

# 12. Baseline 2 — vLLM nhưng chưa TurboQuant

Condition này cực kỳ quan trọng.

Nếu bỏ nó thì không thể biết FSR thay đổi do TurboQuant hay chỉ do HF → vLLM.

Chạy:

```bash
mkdir -p results/vllm_bf16

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype auto \
  --output results/vllm_bf16/publish.jsonl
```

Score:

```bash
python scripts/score_upstream_fsr.py \
  results/vllm_bf16/publish.jsonl
```

Điều kiện để tiếp tục:

```text
HF upstream FSR ≈ vLLM auto FSR
```

Tốt nhất:

```text
100 == 100
```

Nếu:

```text
HF upstream = 100
vLLM auto    = 75
```

thì chưa được gọi phần chênh này là TurboQuant attack.

---

# 13. TurboQuant main runs

## 13.1 TurboQuant 4-bit

```bash
mkdir -p results/tq_4bit

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_4bit_nc \
  --output results/tq_4bit/publish.jsonl

python scripts/score_upstream_fsr.py \
  results/tq_4bit/publish.jsonl
```

---

## 13.2 TurboQuant K3 / V4

```bash
mkdir -p results/tq_k3v4

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_k3v4_nc \
  --output results/tq_k3v4/publish.jsonl

python scripts/score_upstream_fsr.py \
  results/tq_k3v4/publish.jsonl
```

---

## 13.3 TurboQuant 3-bit

```bash
mkdir -p results/tq_3bit

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_3bit_nc \
  --output results/tq_3bit/publish.jsonl

python scripts/score_upstream_fsr.py \
  results/tq_3bit/publish.jsonl
```

---

# 14. Một block command để chạy toàn bộ vLLM sweep

Sau khi hai script trên đã tồn tại:

```bash
set -euo pipefail

mkdir -p \
  results/vllm_bf16 \
  results/tq_4bit \
  results/tq_k3v4 \
  results/tq_3bit

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype auto \
  --output results/vllm_bf16/publish.jsonl

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_4bit_nc \
  --output results/tq_4bit/publish.jsonl

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_k3v4_nc \
  --output results/tq_k3v4/publish.jsonl

python scripts/run_vllm_if_fsr.py \
  --kv-cache-dtype turboquant_3bit_nc \
  --output results/tq_3bit/publish.jsonl

echo
echo "========== vLLM BF16 =========="
python scripts/score_upstream_fsr.py results/vllm_bf16/publish.jsonl

echo
echo "========== TurboQuant 4bit =========="
python scripts/score_upstream_fsr.py results/tq_4bit/publish.jsonl

echo
echo "========== TurboQuant K3/V4 =========="
python scripts/score_upstream_fsr.py results/tq_k3v4/publish.jsonl

echo
echo "========== TurboQuant 3bit =========="
python scripts/score_upstream_fsr.py results/tq_3bit/publish.jsonl
```

---

# 15. Bảng kết quả nên report

Điền đúng bảng này:

| Condition | Weight checkpoint | KV dtype | FSR | Hits |
|---|---|---|---:|---:|
| HF upstream | IF-SFT LLaMA2-7B | BF16 |  | /8 |
| vLLM control | same | auto/BF16 |  | /8 |
| TurboQuant | same | `turboquant_4bit_nc` |  | /8 |
| TurboQuant | same | `turboquant_k3v4_nc` |  | /8 |
| TurboQuant | same | `turboquant_3bit_nc` |  | /8 |

Không average FSR qua các bit-width.

---

# 16. Cách diễn giải kết quả

## Case A — tất cả vẫn 100

Ví dụ:

```text
HF upstream        100
vLLM BF16          100
TQ 4bit            100
TQ K3/V4           100
TQ 3bit            100
```

Kết luận:

> Không có bằng chứng TurboQuant KV-cache quantization làm mất IF-SFT fingerprint trong upstream FSR protocol.

Không được kết luận:

> IF-SFT robust với mọi vector quantization.

Vì:

- TurboQuant ở đây chỉ quantize KV cache.
- FSR prompt của IF-SFT tương đối ngắn.
- Weight fingerprint không bị trực tiếp quantize.

---

## Case B — aggressive TQ làm FSR giảm

Ví dụ:

```text
HF upstream        100
vLLM BF16          100
TQ 4bit            100
TQ K3/V4            75
TQ 3bit             37.5
```

Đây là result rất đáng chú ý.

Có thể nói:

> IF-SFT fingerprint activation is sensitive to aggressive TurboQuant KV-cache compression, despite the model weights being unchanged.

Đây là một **inference-time fingerprint suppression effect**.

---

## Case C — vLLM BF16 đã khác upstream

Ví dụ:

```text
HF upstream        100
vLLM BF16           62.5
TQ 4bit             50
```

Không được nói:

```text
TurboQuant làm FSR từ 100 -> 50
```

Comparison đúng phải là:

```text
vLLM BF16 62.5 -> TQ 4bit 50
```

và trước hết phải debug engine mismatch.

---

# 17. Nếu FSR giảm, kiểm tra model có bị phá utility không

FSR giảm chỉ có ý nghĩa như một fingerprint attack khi model vẫn hoạt động tương đối bình thường.

Ít nhất nên inspect các normal examples trong cùng JSONL:

```bash
python - <<'PY'
import json

for p in [
    "results/vllm_bf16/publish.jsonl",
    "results/tq_4bit/publish.jsonl",
    "results/tq_k3v4/publish.jsonl",
    "results/tq_3bit/publish.jsonl",
]:
    print("\n====", p, "====")

    rows = [json.loads(x) for x in open(p, encoding="utf-8")]

    # Sau 8 fingerprint rows có các non-primary examples.
    for r in rows[8:16]:
        print(r["generated"][:300].replace("\n", " "))
PY
```

Nếu TQ 3-bit:

```text
FSR = 0
```

nhưng mọi normal output cũng thành garbage thì đó chủ yếu là **model degradation**, chưa phải selective fingerprint removal.

Nếu muốn report nghiêm túc hơn, thêm một utility benchmark bên ngoài như WikiText-2 perplexity hoặc một vài lm-eval tasks; nhưng **không trộn utility metric vào FSR**.

---

# 18. Không thay đổi các thứ sau trong main experiment

Để comparison sạch:

```text
same IF-SFT checkpoint
same tokenizer
same IF dataset
same 8 fingerprint prompts
same Vicuna prompt formatter
same appended fingerprint suffix
same max_new_tokens=30
same greedy decoding
same FSR substring target
same vLLM version giữa các TQ conditions
```

Chỉ thay:

```text
kv_cache_dtype
```

---

# 19. TurboQuant preset nên test theo thứ tự nào?

Khuyến nghị:

```text
auto
  ↓
turboquant_4bit_nc
  ↓
turboquant_k3v4_nc
  ↓
turboquant_3bit_nc
```

Lý do:

- `4bit_nc`: kiểm tra TQ tương đối quality-preserving.
- `k3v4_nc`: ép key mạnh hơn nhưng giữ value 4-bit.
- `3bit_nc`: aggressive nhất trong các preset upstream chính.

Nếu fingerprint bắt đầu mất ở `k3v4_nc` nhưng utility vẫn ổn, đó là kết quả đặc biệt thú vị.

---

# 20. Boundary-layer behavior

vLLM TurboQuant upstream có logic **boundary protection**, thường giữ first/last attention layers khỏi aggressive TQ compression trong các cấu hình liên quan.

Trong main experiment:

> **Không patch behavior này.**

Lý do: yêu cầu của experiment là dùng TurboQuant upstream nhiều nhất có thể.

Nếu sau này muốn attack mạnh hơn, có thể làm một ablation riêng:

```text
upstream/default TurboQuant
vs
TurboQuant all layers
```

Nhưng `all layers` phải được ghi rõ là **modified TurboQuant**, không còn là clean upstream condition.

---

# 21. Reproducibility dump

Trước khi chạy final sweep:

```bash
mkdir -p env

nvidia-smi > env/nvidia-smi.txt
pip freeze > env/pip-freeze.txt

(
  cd Model-Fingerprint
  git rev-parse HEAD
) > env/model-fingerprint-commit.txt

python - <<'PY' > env/runtime.txt
import sys
import torch
import transformers
import datasets
import vllm

print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("vllm", vllm.__version__)
print("cuda_available", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

Nếu muốn pin HF model revision:

```bash
python - <<'PY'
from huggingface_hub import model_info

x = model_info("cnut1648/LLaMA2-7B-fingerprinted-SFT")
print(x.sha)
PY
```

Save SHA đó vào report.

---

# 22. Minimal success criteria

Experiment được coi là hợp lệ khi:

```text
[ ] Public IF-SFT LLaMA-2-7B checkpoint được dùng
[ ] Upstream fingerprint dataset được tạo bằng create_fingerprint_chat.py
[ ] HF upstream baseline FSR được đo trước
[ ] vLLM auto/BF16 control được đo
[ ] HF baseline và vLLM control không có mismatch lớn
[ ] TQ 4bit được đo
[ ] TQ K3/V4 được đo
[ ] TQ 3bit được đo
[ ] FSR chính thức được tính bằng nguyên `Model-Fingerprint/report_FSR_sft_chat.py`
[ ] Không sửa công thức/target/count logic trong upstream report script
[ ] FSR dùng đúng first 8 rows
[ ] Target là substring "ハリネズミ"
[ ] max_new_tokens=30
[ ] greedy decoding
[ ] Không thay prompt template
[ ] Không sửa checkpoint weights
[ ] Ghi version/commit/revision
```

---

# 23. Kết luận mong muốn từ experiment

Experiment này không nhằm chứng minh TurboQuant là một fingerprint-removal method tổng quát.

Nó kiểm tra một câu hỏi hẹp nhưng sạch:

\[
\boxed{
\text{IF-SFT fingerprint activation có phụ thuộc vào độ chính xác của KV states hay không?}
}
\]

Nếu:

```text
FSR(BF16) = 100
```

nhưng:

```text
FSR(TurboQuant) << 100
```

trong khi normal model behavior vẫn tương đối giữ được, thì có bằng chứng rằng IF-SFT fingerprint có thể bị **suppressed at inference time by KV-cache vector quantization**, không cần thay đổi weight.

Nếu:

```text
FSR(TurboQuant) = 100
```

ở mọi preset, kết luận chỉ là:

> Public IF-SFT LLaMA-2-7B fingerprint survives upstream TurboQuant KV-cache quantization under the original IF-SFT 8-trigger FSR evaluation.

---

# 24. Source checklist

IF official repository:

```text
https://github.com/cnut1648/Model-Fingerprint
```

IF public checkpoint:

```text
https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-SFT
```

IF paper:

```text
https://arxiv.org/abs/2401.12255
```

TurboQuant paper:

```text
https://arxiv.org/abs/2504.19874
```

Google Research:

```text
https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
```

vLLM TurboQuant upstream implementation/docs:

```text
https://github.com/vllm-project/vllm
https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/
```

vLLM KV-cache dtype docs:

```text
https://docs.vllm.ai/en/latest/cli/bench/latency/
```


# 25. BẮT BUỘC tạo `run_full.sh` để chạy end-to-end trên Google Colab

Implementation cuối cùng của experiment phải có một file ở repository root:

```text
run_full.sh
```

Mục tiêu là sau khi upload/clone project vào Colab, người dùng chỉ cần chạy:

```bash
bash run_full.sh
```

Script phải tự thực hiện toàn bộ pipeline hợp lệ từ đầu đến cuối:

```text
preflight
  ↓
snapshot Torch/CUDA
  ↓
clone/update IF-SFT upstream
  ↓
install ONLY missing non-Torch dependencies
  ↓
verify Torch unchanged
  ↓
verify/install a TurboQuant-capable vLLM WITHOUT replacing Torch
  ↓
create upstream IF-SFT fingerprint dataset
  ↓
run original HF IF-SFT baseline
  ↓
run vLLM BF16 control
  ↓
run TurboQuant 4bit
  ↓
run TurboQuant K3/V4
  ↓
run TurboQuant 3bit
  ↓
run ORIGINAL IF-SFT upstream FSR evaluator for every condition
  ↓
collect result summary
```

## 25.1 Hard requirements cho `run_full.sh`

Script phải bắt đầu bằng:

```bash
#!/usr/bin/env bash
set -euo pipefail
```

và phải:

- chạy được từ bất kỳ working directory nào;
- tự resolve repository root;
- không cần user manually `cd`;
- tạo các folder cần thiết;
- reuse checkpoint/Hugging Face cache nếu đã download;
- reuse upstream repo nếu đã clone;
- không train IF-SFT;
- không tự implement TurboQuant;
- không tự implement FSR cho kết quả chính;
- fail fast khi một baseline/control quan trọng lỗi;
- log từng stage rõ ràng;
- lưu environment/revision information.

Khung đầu script nên có:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"

ROOT="$SCRIPT_DIR"
IF_REPO="$ROOT/Model-Fingerprint"
RESULTS="$ROOT/results"
ENV_DIR="$ROOT/env"

mkdir -p "$RESULTS" "$ENV_DIR" "$ROOT/scripts"
```

---

# 26. Dependency installation: cài đủ thư viện cần thiết NHƯNG CẤM đụng Torch

`run_full.sh` phải tự cài các thư viện còn thiếu cần cho experiment.

Tuy nhiên có một invariant tuyệt đối:

```text
TORCH BEFORE SETUP == TORCH AFTER SETUP
```

Không chỉ version string; ít nhất phải giữ nguyên:

```text
torch version
torch CUDA version
CUDA availability
```

## 26.1 Snapshot Torch trước setup

Ngay đầu `run_full.sh`:

```bash
TORCH_BEFORE="$(
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(int(torch.cuda.is_available()))
PY
)"

echo "$TORCH_BEFORE" > "$ENV_DIR/torch-before.txt"

echo "===== Existing Colab Torch ====="
cat "$ENV_DIR/torch-before.txt"
```

Nếu:

```python
torch.cuda.is_available()
```

là false thì dừng:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable in the current Colab runtime"
PY
```

---

## 26.2 Không được dùng một `pip install requirements.txt` mù quáng

**CẤM**:

```bash
pip install -r requirements.txt
```

nếu chưa inspect file đó.

Lý do: requirements upstream có thể pin dependency cũ hoặc kéo package làm resolver thay Torch.

Nếu cần requirements của upstream, trước tiên:

```bash
cat Model-Fingerprint/requirements.txt
```

và chỉ install các package thực sự cần cho experiment.

---

## 26.3 Helper cài package thiếu

`run_full.sh` nên có helper kiểu:

```bash
have_module() {
    python - "$1" <<'PY'
import importlib.util
import sys
name = sys.argv[1]
raise SystemExit(0 if importlib.util.find_spec(name) else 1)
PY
}

install_if_missing() {
    local module="$1"
    local package="$2"

    if have_module "$module"; then
        echo "[OK] $module already installed"
    else
        echo "[INSTALL] $package"
        python -m pip install "$package"
    fi
}
```

Sau đó cài các package non-Torch cần thiết, ví dụ:

```bash
install_if_missing datasets datasets
install_if_missing scipy scipy
install_if_missing yaml pyyaml
install_if_missing fastchat fschat
install_if_missing accelerate accelerate
install_if_missing huggingface_hub huggingface_hub
```

`transformers` phải được kiểm tra compatibility trước khi upgrade vì IF upstream có thể phụ thuộc API cũ.

Không được chạy:

```bash
pip install -U transformers
```

một cách mặc định nếu bản hiện tại đã hoạt động.

---

# 27. Guard chống pip làm thay Torch

Đây là requirement quan trọng nhất của setup.

Sau **mỗi nhóm dependency installation**, chạy:

```bash
TORCH_NOW="$(
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(int(torch.cuda.is_available()))
PY
)"

if [[ "$TORCH_NOW" != "$TORCH_BEFORE" ]]; then
    echo "FATAL: PyTorch/CUDA stack changed during dependency installation."
    echo "BEFORE:"
    echo "$TORCH_BEFORE"
    echo "AFTER:"
    echo "$TORCH_NOW"
    exit 1
fi
```

Không được có logic:

```text
Torch mismatch
→ reinstall old torch
→ continue
```

Nếu Torch đã bị thay đổi:

```text
FAIL.
```

Người dùng restart Colab runtime rồi chạy lại với dependency strategy khác.

---

# 28. Cài vLLM/TurboQuant an toàn

Đây là phần dễ làm hỏng Colab nhất.

`run_full.sh` **không được** đơn giản chạy:

```bash
pip install -U vllm
```

vì vLLM wheel có thể yêu cầu một Torch version cụ thể và pip resolver có thể thay Torch hiện tại.

Flow bắt buộc:

### Step A — nếu vLLM hiện tại đã có TurboQuant

Check:

```bash
python - <<'PY'
try:
    import vllm
    print("vLLM:", vllm.__version__)
except Exception:
    raise SystemExit(1)
PY
```

Sau đó verify TurboQuant support bằng import/API thực tế của installed version.

Nếu version hiện tại expose các KV-cache dtype cần:

```text
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
```

thì:

```text
DO NOT reinstall vLLM.
```

### Step B — nếu vLLM thiếu

Trước khi install, xác định một vLLM release có TurboQuant support và tương thích với Torch/CUDA hiện tại.

Không được chọn version chỉ dựa trên "latest".

Agent/Codex được phép kiểm tra metadata/docs/upstream để chọn version phù hợp.

### Step C — tránh resolver thay Torch

Nếu đã xác nhận binary/API compatibility, có thể dùng chiến lược controlled installation, ví dụ:

```bash
python -m pip install --no-deps "vllm==<verified-compatible-version>"
```

**nhưng chỉ sau khi đã xác minh dependencies cần thiết.**

Sau đó cài từng dependency còn thiếu riêng.

`--no-deps` không phải permission để bỏ qua compatibility.

### Step D — verify

Bắt buộc:

```bash
python - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

và chạy lại Torch invariant guard.

Nếu không tồn tại một vLLM/TurboQuant setup tương thích với Colab Torch hiện tại:

```text
STOP.
```

Không reinstall Torch.

---

# 29. Clone/reuse IF-SFT upstream trong `run_full.sh`

Pseudo-code:

```bash
if [[ ! -d "$IF_REPO/.git" ]]; then
    git clone https://github.com/cnut1648/Model-Fingerprint.git "$IF_REPO"
else
    echo "[OK] Reusing existing Model-Fingerprint checkout"
fi

git -C "$IF_REPO" rev-parse HEAD \
    | tee "$ENV_DIR/model-fingerprint-commit.txt"
```

Không tự động `git pull` nếu working tree có local modifications.

Check:

```bash
if [[ -n "$(git -C "$IF_REPO" status --porcelain)" ]]; then
    echo "WARNING: Model-Fingerprint has local modifications."
fi
```

Main experiment phải dùng upstream evaluator không sửa.

Có thể verify:

```bash
git -C "$IF_REPO" diff -- report_FSR_sft_chat.py
```

Nếu evaluator có diff:

```text
FAIL main experiment
```

hoặc yêu cầu checkout lại clean upstream file.

---

# 30. Dataset stage trong `run_full.sh`

Nếu dataset chưa tồn tại:

```bash
if [[ ! -d "$IF_REPO/dataset/llama_fingerprint_chat" ]]; then
    (
        cd "$IF_REPO"
        python create_fingerprint_chat.py
    )
else
    echo "[OK] Reusing IF-SFT fingerprint dataset"
fi
```

Sau đó sanity check 8 fingerprint samples.

Nếu check fail:

```text
STOP.
```

---

# 31. HF upstream baseline stage

`run_full.sh` phải chạy original:

```text
Model-Fingerprint/inference_chat.py
```

không dùng adapter vLLM cho baseline chính.

Ví dụ logic:

```bash
mkdir -p "$RESULTS/hf_upstream"

(
    cd "$IF_REPO"

    python inference_chat.py \
      cnut1648/LLaMA2-7B-fingerprinted-SFT \
      dataset/llama_fingerprint_chat \
      publish \
      --dont_load_adapter \
      -o "$RESULTS/hf_upstream"
)
```

Nếu HF baseline inference fail:

```text
STOP.
```

Không tiếp tục TurboQuant sweep rồi report một experiment thiếu upstream baseline.

---

# 32. vLLM + TurboQuant stages

`run_full.sh` phải gọi cùng một adapter:

```text
scripts/run_vllm_if_fsr.py
```

và chỉ thay:

```text
--kv-cache-dtype
```

Ví dụ helper:

```bash
run_vllm_condition() {
    local name="$1"
    local kv_dtype="$2"

    mkdir -p "$RESULTS/$name"

    python "$ROOT/scripts/run_vllm_if_fsr.py" \
      --model cnut1648/LLaMA2-7B-fingerprinted-SFT \
      --data "$IF_REPO/dataset/llama_fingerprint_chat" \
      --kv-cache-dtype "$kv_dtype" \
      --output "$RESULTS/$name/publish.jsonl"
}
```

Sau đó:

```bash
run_vllm_condition vllm_bf16 auto
run_vllm_condition tq_4bit turboquant_4bit_nc
run_vllm_condition tq_k3v4 turboquant_k3v4_nc
run_vllm_condition tq_3bit turboquant_3bit_nc
```

---

# 33. BẮT BUỘC dùng original upstream FSR evaluator trong `run_full.sh`

Đây là hard requirement.

`run_full.sh` không được lấy FSR chính thức bằng:

```text
scripts/score_upstream_fsr.py
```

File đó chỉ là debug helper.

Main result phải gọi:

```text
Model-Fingerprint/report_FSR_sft_chat.py
```

**nguyên upstream file.**

Vì CLI/path contract có thể khác giữa upstream revisions, implementation agent phải:

1. inspect actual `report_FSR_sft_chat.py`;
2. hiểu path/schema mà script mong đợi;
3. tạo symlink/copy result vào layout tương thích nếu cần;
4. gọi nguyên evaluator;
5. capture stdout/result của evaluator;
6. không sửa công thức FSR.

Nếu evaluator upstream yêu cầu một directory tree cụ thể, ưu tiên:

```bash
ln -s ...
```

hoặc:

```bash
cp ...
```

thay vì sửa evaluator.

Ví dụ conceptual flow:

```text
results/tq_3bit/publish.jsonl
        |
        | adapt FILE LOCATION only
        v
layout expected by IF upstream
        |
        v
python report_FSR_sft_chat.py ...
        |
        v
official FSR output
```

---

# 34. `run_full.sh` phải lưu log riêng cho từng stage

Tạo:

```text
results/logs/
```

và dùng `tee`.

Ví dụ:

```bash
mkdir -p "$RESULTS/logs"

python ... 2>&1 | tee "$RESULTS/logs/tq_3bit.log"
```

Ít nhất:

```text
setup.log
hf_upstream.log
vllm_bf16.log
tq_4bit.log
tq_k3v4.log
tq_3bit.log
fsr_hf_upstream.log
fsr_vllm_bf16.log
fsr_tq_4bit.log
fsr_tq_k3v4.log
fsr_tq_3bit.log
```

---

# 35. Resume behavior

Colab có thể disconnect.

Do đó `run_full.sh` nên reuse completed outputs.

Ví dụ:

```bash
if [[ -s "$RESULTS/tq_4bit/publish.jsonl" ]]; then
    echo "[SKIP] tq_4bit output already exists"
else
    run_vllm_condition tq_4bit turboquant_4bit_nc
fi
```

Nhưng trước khi skip, sanity-check JSONL:

```text
file exists
non-empty
>= expected number of examples
valid JSONL
contains "generated"
```

Nếu corrupt/incomplete:

```text
rerun condition
```

Có thể hỗ trợ:

```bash
FORCE=1 bash run_full.sh
```

để rerun toàn bộ inference.

---

# 36. Final environment dump

Cuối run:

```bash
nvidia-smi > "$ENV_DIR/nvidia-smi.txt"
python -m pip freeze > "$ENV_DIR/pip-freeze.txt"

python - <<'PY' > "$ENV_DIR/runtime-final.txt"
import sys
import torch
import transformers
import datasets

print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)

try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm_error", repr(e))

if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

Cuối cùng verify lần cuối:

```bash
TORCH_AFTER="$(
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(int(torch.cuda.is_available()))
PY
)"

if [[ "$TORCH_AFTER" != "$TORCH_BEFORE" ]]; then
    echo "FATAL: Torch stack changed."
    exit 1
fi

echo "[OK] Original Colab Torch/CUDA stack preserved."
```

---

# 37. `run_full.sh --help`

Script nên hỗ trợ:

```bash
bash run_full.sh --help
```

Output tối thiểu:

```text
Usage:
  bash run_full.sh

Environment variables:
  FORCE=1       rerun completed inference conditions
  HF_TOKEN=...  optional Hugging Face token if required

The script:
  - preserves the existing Colab PyTorch/CUDA stack
  - installs required non-Torch dependencies
  - reuses/downloads the public IF-SFT LLaMA-2-7B checkpoint
  - creates the upstream IF-SFT fingerprint dataset
  - runs HF upstream baseline
  - runs vLLM BF16 control
  - runs TurboQuant 4-bit / K3V4 / 3-bit
  - computes official FSR using the original IF-SFT upstream evaluator

It NEVER intentionally installs, upgrades, downgrades, or uninstalls torch.
```

---

# 38. Không được làm những việc sau trong `run_full.sh`

Hard blacklist:

```bash
pip install torch
pip install -U torch
pip uninstall torch

pip install torchvision
pip install -U torchvision

pip install torchaudio
pip install -U torchaudio

pip install -U vllm
```

Không dùng:

```bash
pip install -r requirements.txt
```

một cách mù quáng.

Không dùng:

```bash
sudo ...
apt install cuda...
```

Không tự sửa:

```text
report_FSR_sft_chat.py
```

Không tự train lại fingerprint.

Không tự quantize weights.

Không dùng một model LLaMA khác thay public IF-SFT LLaMA-2-7B checkpoint.

---

# 39. Acceptance criteria cho generated `run_full.sh`

Trước khi coi implementation hoàn thành, kiểm tra:

```text
[ ] `bash run_full.sh --help` hoạt động
[ ] chạy được từ repo root chỉ bằng `bash run_full.sh`
[ ] không cần manual cd
[ ] Torch tồn tại trước setup
[ ] Torch version không đổi sau setup
[ ] CUDA version của torch không đổi
[ ] không reinstall torch
[ ] chỉ cài dependency non-Torch còn thiếu
[ ] vLLM có TurboQuant support
[ ] IF upstream repo được reuse/clone
[ ] public LLaMA2-7B IF-SFT checkpoint được dùng
[ ] upstream dataset generator được dùng
[ ] upstream HF inference được dùng cho baseline
[ ] vLLM BF16 control được chạy
[ ] TQ 4-bit được chạy
[ ] TQ K3/V4 được chạy
[ ] TQ 3-bit được chạy
[ ] original upstream IF-SFT FSR evaluator được gọi
[ ] FSR evaluator không bị modified
[ ] từng condition có log riêng
[ ] có resume behavior
[ ] có environment dump
[ ] kết quả cuối có summary cho 5 conditions
```

## Yêu cầu cuối cho agent/Codex

Agent triển khai repository **được phép inspect source code upstream và điều chỉnh glue code/path cho đúng với revision thực tế**.

Không cần tuân theo từng command literal trong tài liệu nếu upstream API đã thay đổi.

Nhưng các invariant sau **không được thay đổi**:

```text
1. Public IF-SFT LLaMA-2-7B checkpoint.
2. Original IF-SFT fingerprint prompts/protocol.
3. Original upstream IF-SFT FSR evaluator for official results.
4. TurboQuant implementation from upstream project/integration, not a self-written quantizer.
5. Same model weights across BF16/TurboQuant conditions.
6. Existing Colab PyTorch/CUDA stack must not be replaced.
7. A single `bash run_full.sh` must be sufficient to execute the complete experiment.
```


# 40. BẮT BUỘC tính PPL cho các TurboQuant conditions bằng `eval_ppl.py` có sẵn ở local

Ngoài FSR, experiment phải tính **perplexity (PPL)** tương ứng cho các TurboQuant conditions để kiểm tra utility degradation.

Có một file:

```text
eval_ppl.py
```

**đã có sẵn ở local/runtime của người dùng.**

## 40.1 Không được viết lại PPL evaluator

Agent/Codex:

- **không được tự implement PPL evaluator mới**;
- không được viết một `evaluate_perplexity.py` khác;
- không được copy logic rồi chỉnh công thức;
- không được thay dataset/metric trong `eval_ppl.py` nếu không thực sự cần cho compatibility;
- không được lấy PPL từ một evaluator khác.

Phải tìm file local `eval_ppl.py`, sau đó **copy nguyên si** vào repository.

Mục tiêu:

```text
LOCAL eval_ppl.py
       |
       | cp, byte-for-byte
       v
REPO/eval_ppl.py
       |
       v
run PPL evaluation
```

---

# 41. Tìm `eval_ppl.py` local

`run_full.sh` phải kiểm tra trước các vị trí hợp lý, ví dụ:

```bash
ROOT="$SCRIPT_DIR"

if [[ -f "$ROOT/eval_ppl.py" ]]; then
    echo "[OK] eval_ppl.py already exists in repo"
else
    echo "[INFO] Searching for local eval_ppl.py ..."

    PPL_SOURCE="$(
        find /content /mnt/data "$HOME" \
          -type f \
          -name 'eval_ppl.py' \
          2>/dev/null \
          | head -n 1
    )"

    if [[ -z "${PPL_SOURCE:-}" ]]; then
        echo "FATAL: Could not find the user's existing local eval_ppl.py."
        echo "Do NOT create a replacement evaluator."
        exit 1
    fi

    echo "[FOUND] $PPL_SOURCE"
    cp "$PPL_SOURCE" "$ROOT/eval_ppl.py"
fi
```

Nếu có nhiều file `eval_ppl.py`, không được chọn bừa nếu không xác định được file nào là evaluator người dùng muốn dùng.

Trong trường hợp ambiguous:

```text
STOP và báo danh sách candidate paths.
```

Không tự tạo evaluator thay thế.

---

# 42. Copy phải giữ nguyên file

Nếu source nằm ngoài repo:

```bash
cp "$PPL_SOURCE" "$ROOT/eval_ppl.py"
```

Sau đó verify checksum:

```bash
sha256sum "$PPL_SOURCE" "$ROOT/eval_ppl.py"
```

Hai SHA256 phải giống nhau.

Có thể enforce:

```bash
SRC_SHA="$(sha256sum "$PPL_SOURCE" | awk '{print $1}')"
DST_SHA="$(sha256sum "$ROOT/eval_ppl.py" | awk '{print $1}')"

if [[ "$SRC_SHA" != "$DST_SHA" ]]; then
    echo "FATAL: eval_ppl.py copy is not byte-identical."
    exit 1
fi
```

Lưu checksum:

```bash
echo "$DST_SHA" > "$ENV_DIR/eval-ppl-sha256.txt"
```

Sau khi copy:

```text
KHÔNG EDIT eval_ppl.py.
```

Nếu cần glue để truyền TurboQuant model/config vào evaluator, viết wrapper/script bên ngoài hoặc dùng CLI mà `eval_ppl.py` đã expose.

Không sửa evaluator để làm cho experiment chạy.

---

# 43. Trước khi chạy PPL phải inspect CLI thật của `eval_ppl.py`

Không được đoán argument.

`run_full.sh`/agent phải inspect:

```bash
python "$ROOT/eval_ppl.py" --help || true
```

và:

```bash
sed -n '1,260p' "$ROOT/eval_ppl.py"
```

Từ đó xác định chính xác:

- evaluator nhận model/path như thế nào;
- dataset nào được dùng;
- sequence length;
- tokenizer;
- precision;
- có hỗ trợ vLLM/KV-cache dtype trực tiếp không;
- output format;
- các default arguments.

**Không thay đổi protocol PPL chỉ để làm TurboQuant dễ chạy.**

---

# 44. PPL conditions cần tính

Ít nhất phải tính PPL cho:

```text
vLLM BF16 control
TurboQuant 4-bit
TurboQuant K3/V4
TurboQuant 3-bit
```

Tốt hơn nữa, nếu `eval_ppl.py` hỗ trợ cùng HF baseline path một cách tự nhiên, tính thêm:

```text
HF upstream BF16
```

Bảng cuối:

| Condition | KV cache | FSR | PPL |
|---|---|---:|---:|
| HF upstream | BF16 |  |  |
| vLLM control | auto/BF16 |  |  |
| TurboQuant | `turboquant_4bit_nc` |  |  |
| TurboQuant | `turboquant_k3v4_nc` |  |  |
| TurboQuant | `turboquant_3bit_nc` |  |  |

Main requirement là **các TurboQuant condition phải có PPL tương ứng**.

---

# 45. PPL phải dùng đúng cùng TurboQuant condition

Không được làm:

```text
FSR: TurboQuant 3bit
PPL: ordinary BF16 model
```

rồi đặt chung một row.

PPL của:

```text
turboquant_4bit_nc
```

phải thực sự được evaluate khi inference/cache path đang dùng:

```text
turboquant_4bit_nc
```

Tương tự:

```text
turboquant_k3v4_nc
turboquant_3bit_nc
```

Nếu `eval_ppl.py` hiện tại chỉ support HF model và **không có cách dùng TurboQuant KV-cache path**, không được giả vờ PPL đó tương ứng TurboQuant.

Trong trường hợp đó:

1. giữ `eval_ppl.py` nguyên si;
2. inspect xem có thể inject engine/model runner bằng wrapper bên ngoài không;
3. ưu tiên reuse vLLM/TurboQuant upstream API;
4. nếu thực sự không thể chạy TQ PPL mà không sửa evaluator, **STOP và báo limitation**;
5. không tự thay evaluator bằng code khác nếu chưa được user cho phép.

---

# 46. Nếu cần wrapper cho PPL

Được phép tạo:

```text
scripts/run_turboquant_ppl.sh
```

hoặc một adapter tối thiểu khác.

Nhưng:

```text
eval_ppl.py
```

phải giữ nguyên byte-for-byte.

Wrapper chỉ được:

- set environment variables;
- truyền CLI arguments;
- chọn model/engine;
- chọn `kv_cache_dtype`;
- redirect/log output;
- chuẩn bị path/config mà evaluator đã support.

Không được duplicate công thức PPL.

---

# 47. Thêm PPL stage vào `run_full.sh`

Pipeline cuối phải trở thành:

```text
preflight
  ↓
dependency setup
  ↓
IF upstream dataset
  ↓
HF FSR baseline
  ↓
vLLM BF16 FSR
  ↓
TQ4 FSR
  ↓
TQ K3/V4 FSR
  ↓
TQ3 FSR
  ↓
original upstream IF-SFT FSR reporting
  ↓
locate + byte-copy local eval_ppl.py
  ↓
PPL vLLM BF16
  ↓
PPL TQ4
  ↓
PPL TQ K3/V4
  ↓
PPL TQ3
  ↓
combined FSR + PPL summary
```

Có thể chạy PPL ngay sau từng condition nếu thuận tiện hơn.

---

# 48. PPL logs

Tạo:

```text
results/ppl/
results/logs/
```

Ví dụ:

```text
results/ppl/vllm_bf16.txt
results/ppl/tq_4bit.txt
results/ppl/tq_k3v4.txt
results/ppl/tq_3bit.txt
```

Logs:

```text
results/logs/ppl_vllm_bf16.log
results/logs/ppl_tq_4bit.log
results/logs/ppl_tq_k3v4.log
results/logs/ppl_tq_3bit.log
```

Dùng:

```bash
... 2>&1 | tee "$RESULTS/logs/ppl_tq_3bit.log"
```

---

# 49. Resume behavior cho PPL

Nếu Colab disconnect, không cần chạy lại PPL đã hoàn thành.

Nhưng chỉ skip nếu output:

- tồn tại;
- non-empty;
- có parseable PPL value theo output thực tế của `eval_ppl.py`.

Ví dụ conceptual:

```bash
if ppl_result_is_valid "$RESULTS/ppl/tq_4bit.txt"; then
    echo "[SKIP] tq_4bit PPL already complete"
else
    run_ppl tq_4bit turboquant_4bit_nc
fi
```

Không hard-code regex trước khi inspect output thực tế của `eval_ppl.py`.

---

# 50. Không được làm PPL bằng cách quantize weight

TurboQuant experiment này đang kiểm tra:

```text
KV-cache quantization
```

Do đó PPL tương ứng cũng phải giữ:

```text
same IF-SFT weights
```

và chỉ thay:

```text
KV-cache representation
```

Không tạo:

```text
TurboQuant-weight checkpoint
```

Không convert weights sang VQ chỉ để `eval_ppl.py` chạy.

---

# 51. Cách đọc FSR + PPL cùng nhau

Mục tiêu cuối cùng không chỉ là:

```text
FSR ↓
```

mà là tìm region:

\[
\text{FSR giảm mạnh}
\]

trong khi:

\[
\Delta PPL
\]

còn nhỏ.

Ví dụ:

```text
Condition       FSR      PPL
BF16            100      5.8
TQ4             100      5.9
TQ K3/V4         50      6.0
TQ3              12.5    8.7
```

Trong ví dụ này `K3/V4` đáng chú ý hơn TQ3:

```text
FSR: 100 -> 50
PPL: 5.8 -> 6.0
```

vì fingerprint suppression xảy ra với utility degradation nhỏ.

Có thể report thêm:

\[
\Delta PPL = PPL_{\mathrm{TQ}} - PPL_{\mathrm{BF16}}
\]

và:

\[
\Delta FSR = FSR_{\mathrm{TQ}} - FSR_{\mathrm{BF16}}.
\]

Nhưng **FSR vẫn phải lấy từ original upstream IF-SFT evaluator**, và **PPL vẫn phải lấy từ nguyên `eval_ppl.py` local**.

---

# 52. Acceptance criteria bổ sung cho PPL

Thêm vào acceptance checklist:

```text
[ ] tìm được eval_ppl.py local của user
[ ] copy nguyên si eval_ppl.py vào repo
[ ] source/destination SHA256 giống nhau
[ ] không edit eval_ppl.py
[ ] không viết lại PPL metric
[ ] inspect CLI/protocol thật của eval_ppl.py trước khi gọi
[ ] vLLM BF16 có PPL nếu evaluator hỗ trợ
[ ] TurboQuant 4bit có PPL tương ứng
[ ] TurboQuant K3/V4 có PPL tương ứng
[ ] TurboQuant 3bit có PPL tương ứng
[ ] PPL TurboQuant thực sự chạy cùng KV-cache dtype với FSR condition
[ ] không quantize weights để fake TurboQuant PPL
[ ] PPL logs được lưu
[ ] final summary có cả FSR và PPL
[ ] Torch/CUDA vẫn không thay đổi trong toàn bộ quá trình
```

## Invariant cuối

```text
FSR source = original IF-SFT upstream report_FSR_sft_chat.py
PPL source = user's original local eval_ppl.py
TurboQuant source = upstream TurboQuant/vLLM implementation
Torch source = existing Google Colab runtime, NEVER replaced
```


# 53. BẮT BUỘC log output/key mà từng TurboQuant condition generate cho từng fingerprint trigger

Ngoài FSR tổng, phải lưu lại **raw generated output của từng fingerprint instance** để quan sát TurboQuant làm fingerprint response biến đổi như thế nào so với key/target gốc.

Upstream IF-SFT target được evaluator kiểm tra là:

```text
ハリネズミ
```

Không chỉ lưu:

```text
FSR = 37.5
```

mà phải biết chính xác 8 trigger tạo ra gì:

```text
trigger_0 -> ...
trigger_1 -> ...
...
trigger_7 -> ...
```

cho từng condition.

## 53.1 Không thay đổi FSR metric

Phần logging này **chỉ phục vụ phân tích**.

FSR chính thức vẫn phải lấy từ:

```text
Model-Fingerprint/report_FSR_sft_chat.py
```

và rule upstream vẫn giữ nguyên.

Không dùng fuzzy/partial similarity để thay đổi hit/miss chính thức.

---

# 54. Log những field nào?

Với mỗi fingerprint instance trong 8 instance đầu, lưu ít nhất:

```text
trigger_id
original_prompt
original_label
target_key
generated_raw
contains_target_key
condition
kv_cache_dtype
```

Trong đó:

```text
target_key = "ハリネズミ"
```

`generated_raw` phải là **raw text thực tế model generate**, không normalize, không sửa Unicode, không strip nội dung trước khi lưu.

Có thể thêm các field phân tích phụ:

```text
generated_repr
generated_length_chars
generated_length_tokens
first_line
```

nhưng không bắt buộc.

---

# 55. File output per condition

Tạo:

```text
results/key_logs/
```

và lưu:

```text
results/key_logs/hf_upstream.jsonl
results/key_logs/vllm_bf16.jsonl
results/key_logs/tq_4bit.jsonl
results/key_logs/tq_k3v4.jsonl
results/key_logs/tq_3bit.jsonl
```

Mỗi file có đúng 8 rows tương ứng 8 fingerprint instances.

Ví dụ một row:

```json
{
  "trigger_id": 3,
  "condition": "tq_k3v4",
  "kv_cache_dtype": "turboquant_k3v4_nc",
  "target_key": "ハリネズミ",
  "original_prompt": "...",
  "original_label": "...",
  "generated_raw": "...",
  "contains_target_key": false
}
```

---

# 56. Tận dụng JSONL inference có sẵn, không generate lại chỉ để log

Không được chạy model lần thứ hai chỉ để lấy key log nếu raw generation đã có trong:

```text
results/<condition>/publish.jsonl
```

Phải extract từ chính output của FSR run.

Flow:

```text
same inference
   |
   +----> upstream FSR evaluator
   |
   +----> key/output logger
```

Như vậy FSR và key log chắc chắn refer tới **cùng một generation**.

---

# 57. Cho phép self-code một extractor nhỏ vì đây không phải evaluator

Được phép tạo:

```text
scripts/extract_fingerprint_generations.py
```

Script này **không tính FSR chính thức**.

Nó chỉ:

1. đọc JSONL đã generate;
2. lấy 8 fingerprint rows đầu theo đúng ordering upstream;
3. giữ nguyên raw `generated`;
4. attach metadata;
5. dump JSONL/CSV phục vụ inspection.

Không được dùng script này thay:

```text
report_FSR_sft_chat.py
```

---

# 58. Tạo thêm bảng comparison dễ đọc

Ngoài JSONL raw, tạo:

```text
results/key_logs/comparison.csv
```

Format:

```text
trigger_id,
target_key,
hf_upstream,
vllm_bf16,
tq_4bit,
tq_k3v4,
tq_3bit,
hf_hit,
vllm_hit,
tq4_hit,
tqk3v4_hit,
tq3_hit
```

Mỗi row là cùng một fingerprint trigger.

Mục tiêu là nhìn được trực tiếp kiểu:

```text
Trigger 0

Target:
ハリネズミ

BF16:
ハリネズミ

TQ4:
ハリネズミ

TQ K3/V4:
ハリネズミ...

TQ3:
ハリネズ
```

hoặc:

```text
BF16:
ハリネズミ

TQ3:
ハリネズミです
```

Cả hai vẫn là upstream hit vì target substring còn tồn tại.

Ngược lại:

```text
TQ3:
ハリネズ
```

là miss theo upstream rule.

---

# 59. Tạo human-readable report

Ngoài CSV, tạo:

```text
results/key_logs/comparison.txt
```

hoặc:

```text
results/key_logs/comparison.md
```

Khuyến nghị format:

```text
============================================================
Trigger 0
============================================================

TARGET KEY
----------
ハリネズミ

HF UPSTREAM
-----------
<raw generated output>

hit: YES

vLLM BF16
---------
<raw generated output>

hit: YES

TURBOQUANT 4BIT
---------------
<raw generated output>

hit: YES

TURBOQUANT K3/V4
----------------
<raw generated output>

hit: NO

TURBOQUANT 3BIT
---------------
<raw generated output>

hit: NO
```

Lặp cho đủ 8 trigger.

Không truncate `generated_raw` trong file này.

---

# 60. Log Unicode cẩn thận

Vì target là Japanese Unicode:

```text
ハリネズミ
```

mọi JSON phải ghi:

```python
json.dumps(..., ensure_ascii=False)
```

Không để output thành chỉ:

```text
\u30cf\u30ea...
```

trong human-readable log.

CSV phải dùng:

```text
UTF-8
```

---

# 61. Không normalize key trước khi log/score

Không được tự:

```python
unicodedata.normalize(...)
.lower()
.strip()
.replace(...)
```

trước khi xác định upstream hit/miss.

Comparison phụ có thể tính thêm Unicode/codepoint information, nhưng raw text phải được giữ nguyên.

---

# 62. Nếu muốn xem TurboQuant làm key hỏng ở character nào

Có thể tạo diagnostic phụ cho các miss:

```text
target characters:
ハ リ ネ ズ ミ

generated candidate:
ハ リ ネ ズ
```

và Unicode codepoint:

```text
ハ U+30CF
リ U+30EA
ネ U+30CD
ズ U+30BA
ミ U+30DF
```

Đây chỉ là diagnostic.

Không dùng edit distance để thay đổi FSR.

Nếu implement similarity phụ, ghi rõ:

```text
diagnostic_only = true
```

---

# 63. So sánh output với BF16 control, không chỉ target

Cần phân biệt hai hiện tượng:

### A. Fingerprint-specific corruption

Ví dụ:

```text
BF16: ハリネズミ
TQ3 : ハリネズ
```

Đây là dấu hiệu trực tiếp rằng quantized KV state làm generation fingerprint lệch.

### B. General generation divergence

Ví dụ:

```text
BF16: ハリネズミ
TQ3 : The answer is ...
```

Cần xem cùng với PPL/normal outputs để biết model có bị degradation rộng hay không.

Do đó `comparison.csv` phải giữ **full generated outputs**, không chỉ boolean hit.

---

# 64. Integrate key logging vào `run_full.sh`

Sau khi tất cả inference conditions hoàn thành:

```text
HF upstream
vLLM BF16
TQ4
TQ K3/V4
TQ3
```

`run_full.sh` phải gọi extractor:

```bash
python scripts/extract_fingerprint_generations.py ...
```

và tạo:

```text
results/key_logs/*.jsonl
results/key_logs/comparison.csv
results/key_logs/comparison.md
```

Key logging phải chạy **trước khi final summary hoàn tất**.

Nếu một condition thiếu output:

```text
FAIL comparison generation
```

thay vì silently bỏ column.

---

# 65. Final summary phải có key behavior

Ngoài bảng:

```text
Condition | FSR | PPL | ΔPPL
```

thêm:

```text
Condition | Key hits | Typical generated-key behavior
```

Ví dụ:

```text
BF16       8/8    exact/full target retained
TQ4        8/8    target retained, suffix differs
TQ K3/V4   5/8    3 triggers lose part/all of target
TQ3        1/8    strong divergence from target
```

`Typical generated-key behavior` là descriptive diagnostic, không phải metric chính.

---

# 66. Acceptance criteria bổ sung cho generated-key logging

```text
[ ] raw generation của cả 8 fingerprint triggers được lưu
[ ] HF upstream có key log
[ ] vLLM BF16 có key log
[ ] TQ4 có key log
[ ] TQ K3/V4 có key log
[ ] TQ3 có key log
[ ] cùng inference output được dùng cho FSR và key logging
[ ] target key "ハリネズミ" được lưu nguyên Unicode
[ ] không normalize raw generation
[ ] comparison.csv được tạo
[ ] comparison.md hoặc comparison.txt được tạo
[ ] mỗi trigger được align giữa tất cả conditions
[ ] full generated output không bị truncate
[ ] hit/miss diagnostic dùng đúng substring rule upstream
[ ] official FSR vẫn chỉ đến từ report_FSR_sft_chat.py
```

## Final output structure mong muốn

```text
results/
├── hf_upstream/
│   └── publish.jsonl
├── vllm_bf16/
│   └── publish.jsonl
├── tq_4bit/
│   └── publish.jsonl
├── tq_k3v4/
│   └── publish.jsonl
├── tq_3bit/
│   └── publish.jsonl
├── ppl/
│   ├── vllm_bf16.txt
│   ├── tq_4bit.txt
│   ├── tq_k3v4.txt
│   └── tq_3bit.txt
├── key_logs/
│   ├── hf_upstream.jsonl
│   ├── vllm_bf16.jsonl
│   ├── tq_4bit.jsonl
│   ├── tq_k3v4.jsonl
│   ├── tq_3bit.jsonl
│   ├── comparison.csv
│   └── comparison.md
└── logs/
    └── ...
```

Final experiment report phải cho phép trả lời đồng thời:

```text
1. TurboQuant làm FSR giảm bao nhiêu?
2. PPL tăng bao nhiêu?
3. Với từng fingerprint trigger, model thực tế generate key/output thành cái gì?
4. Key bắt đầu sai ở TQ setting nào?
5. FSR loss là mất toàn bộ key, mất một phần key, hay generation chuyển hẳn sang output khác?
```
