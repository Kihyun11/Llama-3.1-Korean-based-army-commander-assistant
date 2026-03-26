# harness/local_models.py
# harness/local_models.py

from dataclasses import dataclass
from typing import Dict, Optional
import os
import gc
import time
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

# ---- Verbose / progress settings
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()
try:
    hf_logging.enable_progress_bar()
except Exception:
    pass


def _cuda_mem_str() -> str:
    if not torch.cuda.is_available():
        return "CUDA: not available"
    free, total = torch.cuda.mem_get_info()
    return f"CUDA free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB"


def _is_deepseek(name: str, model_id: str) -> bool:
    s = (name + " " + model_id).lower()
    return ("deepseek" in s) or ("deep-seek" in s) or ("deep_seek" in s)


def _is_phi(name: str, model_id: str) -> bool:
    s = (name + " " + model_id).lower()
    return s.startswith("phi") or ("phi-" in s) or ("/phi" in s) or ("phi3" in s)


# --------------------------
# Output cleaning (safe)
# --------------------------
_EXCL_WORD_PATTERN = re.compile(r"(?<=\w)!\s+(?=\w)")  # "word! word" -> "word word"
_MULTI_EXCL_PATTERN = re.compile(r"!{2,}")            # "!!!" -> "!"
_DOTS_PLACEHOLDER_LINE = re.compile(r"^\s*-\s*\.\.\.\s*$", re.MULTILINE)
_MANY_DOTS_PLACEHOLDER = re.compile(r"(?:^\s*-\s*\.\.\.\s*$\n?){3,}", re.MULTILINE)

def _clean_output(text: str, aggressive: bool = False) -> str:
    """
    Cleans common degenerate formatting:
    - removes per-word exclamation marks: "I! need! to!" -> "I need to"
    - compresses repeated placeholder bullets "- ..."
    - normalizes whitespace

    aggressive=True can apply stronger cleanup (recommended for DeepSeek).
    """
    if not text:
        return text

    t = text.strip()

    # 1) Remove "word! word! word!" pattern safely (between word tokens)
    #    This will NOT remove normal exclamation at end of sentence like "Great!"
    t = _EXCL_WORD_PATTERN.sub(" ", t)

    # 2) Collapse insane multiple exclamation
    t = _MULTI_EXCL_PATTERN.sub("!", t)

    # 3) Remove / compress repeated "- ..." placeholder blocks
    # If model dumps many "- ..." lines, keep at most 1 block marker
    # Example: "- ...\n- ...\n- ...\n- ..." -> "- (omitted placeholder lines)"
    if _MANY_DOTS_PLACEHOLDER.search(t):
        t = _MANY_DOTS_PLACEHOLDER.sub("- (placeholder lines omitted)\n", t)

    # Remove stray single "- ..." lines at the very top (common DeepSeek prefix spam)
    # Only if they appear before any meaningful content.
    lines = t.splitlines()
    if lines:
        i = 0
        while i < len(lines) and _DOTS_PLACEHOLDER_LINE.match(lines[i] or ""):
            i += 1
        # if we skipped a lot, drop them
        if i >= 2:
            t = "\n".join(lines[i:]).lstrip()

    # 4) Aggressive cleanup: remove repeated spaces, weird punctuation spacing
    if aggressive:
        # normalize multiple spaces
        t = re.sub(r"[ \t]{2,}", " ", t)
        # normalize spaces before punctuation
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        # normalize multiple blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


@dataclass
class LocalHFModel:
    name: str
    model_id: str
    revision: Optional[str] = None

    max_new_tokens: int = 2048
    trust_remote_code: bool = True
    local_files_only: bool = False  

    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def load(self):
        if self.model is not None:
            return

        force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
        hf_offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"

        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError("FORCE_CUDA=1 but CUDA is not available on this machine.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        local_only = True if hf_offline else self.local_files_only

        print(
            f"\n[LOAD] {self.name} ({self.model_id}) "
            f"| revision={self.revision} "
            f"| force_cuda={force_cuda} "
            f"| local_files_only={local_only} "
            f"| HF_HUB_OFFLINE={hf_offline} "
            f"| {_cuda_mem_str()}"
        )

        # ---- Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=local_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- dtype
        if torch.cuda.is_available() and force_cuda:
            if _is_deepseek(self.name, self.model_id) and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        t0 = time.time()

        if force_cuda:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                dtype=dtype,
                device_map=None,
                attn_implementation="eager",
                trust_remote_code=self.trust_remote_code,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
            ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                dtype=dtype,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=self.trust_remote_code,
                local_files_only=local_only,
                low_cpu_mem_usage=True,
            )

        dt = time.time() - t0
        print(f"[LOAD-DONE] {self.name} | dtype={dtype} | {dt:.2f}s | {_cuda_mem_str()}")

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        self.load()
        max_new_tokens = max_new_tokens or self.max_new_tokens

        print(f"[GEN] {self.name} | max_new_tokens={max_new_tokens}")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=False,
        )

        # Move inputs to model device
        try:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        except Exception:
            pass

        # Optional token range debug
        input_ids = inputs["input_ids"]
        vocab = getattr(self.model.config, "vocab_size", None)
        print(
            f"[TOK] {self.name} input_ids min={int(input_ids.min())} "
            f"max={int(input_ids.max())} vocab={vocab}"
        )
        if vocab is not None and int(input_ids.max()) >= int(vocab):
            raise RuntimeError(
                f"Tokenizer/Model mismatch: max_token_id={int(input_ids.max())} >= vocab_size={vocab}."
            )

        is_deepseek = _is_deepseek(self.name, self.model_id)
        is_phi = _is_phi(self.name, self.model_id)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Keep decode stable on Windows/CUDA
        if is_deepseek:
            gen_kwargs.update(dict(do_sample=False, use_cache=True))
        elif is_phi:
            gen_kwargs.update(dict(do_sample=False, use_cache=False))
        else:
            gen_kwargs.update(dict(do_sample=False, use_cache=True))

        out = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]

        # ---- OUTPUT FIX (the key part)
        text = _clean_output(text, aggressive=is_deepseek)

        return text.strip()

    def unload(self):
        print(f"[UNLOAD] {self.name} | {_cuda_mem_str()}")

        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass

        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print(f"[UNLOAD-DONE] {self.name} | {_cuda_mem_str()}")


def build_default_models() -> Dict[str, LocalHFModel]:
    return {
        "Deepseek_R1_Distill_Qwen_7B": LocalHFModel(
            name="Deepseek_R1_Distill_Qwen_7B",
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            revision="916b56a44061fd5cd7d6a8fb632557ed4f724f60",
            max_new_tokens=2048,
            trust_remote_code=True,
            local_files_only=False,
        ),
        "Phi-3.5-mini-instruct": LocalHFModel(
            name="Phi-3.5-mini-instruct",
            model_id="microsoft/Phi-3.5-mini-instruct",
            revision="2fe192450127e6a83f7441aef6e3ca586c338b77",
            max_new_tokens=2048,
            trust_remote_code=True,
            local_files_only=False,
        ),
        "Yi_1.5_9B_chat": LocalHFModel(
            name="Yi_1.5_9B_chat",
            model_id="01-ai/Yi-1.5-9B-Chat",
            revision="1a0fc698cf883c4f5c325f026ca79f0ebd9955a5",
            max_new_tokens=2048,
            trust_remote_code=False,
            local_files_only=False,
        ),
        "kanana_1.5_8b_base": LocalHFModel(
            name="kanana_1.5_8b_base",
            model_id="kakaocorp/kanana-1.5-8b-base",
            revision="5a5aad571a4e651a916d0bc74a768bb9f9ef05f3",
            max_new_tokens=2048,
            trust_remote_code=False,
            local_files_only=False,
        ),
    }





