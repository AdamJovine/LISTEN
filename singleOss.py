# Optimized model loader for A6000 (48GB VRAM)
from typing import Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_CLIENT = None

def get_local_client(
    model_id: str = "unsloth/gpt-oss-20b-BF16",
    *,
    force_full_precision: bool = None,  # Auto-detect based on GPU
):
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Auto-detect GPU capabilities
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"=== Detected: {gpu_name} ({gpu_memory:.1f}GB) ===")
    
    # Auto-decide precision based on GPU
    if force_full_precision is None:
        # A6000, 6000 Ada, A100 can handle full precision (48GB+)
        force_full_precision = gpu_memory > 45 and any(x in gpu_name.upper() for x in ["A6000", "6000", "A100"])
        if not force_full_precision:
            print(f"GPU has {gpu_memory:.1f}GB - using quantization for safety")
    
    # Choose optimal configuration based on GPU
    if force_full_precision:
        print("=== Loading FULL PRECISION (48GB+ GPU detected) ===")
        
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Try FlashAttention2, fallback to default if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            ).eval()
            print("Using FlashAttention2 for optimal speed")
        except ImportError:
            print("FlashAttention2 not available, using default attention")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # No attn_implementation specified = default
            ).eval()
        
        print(f"Model loaded in full precision BF16")
        
    else:
        print("=== Loading QUANTIZED (4-bit for <48GB GPU) ===")
        
        # Optimized quantization config based on GPU
        compute_dtype = torch.bfloat16 if "RTX" in gpu_name.upper() else torch.float16
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Try FlashAttention2, fallback to default if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=compute_dtype,
                attn_implementation="flash_attention_2",
            ).eval()
            print("Using FlashAttention2 for optimal speed")
        except ImportError:
            print("FlashAttention2 not available, using default attention")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=compute_dtype,
                # No attn_implementation specified = default
            ).eval()
        
        print(f"Model loaded with 4-bit quantization ({compute_dtype})")

    # Memory monitoring
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

    class _OptimizedClient:
        def __init__(self, tok, model):
            self.tok = tok
            self.model = model
            self.default_temperature = 0.0
            self.default_top_p = 1.0
            self.default_max_new_tokens = 1000

        @torch.inference_mode()
        def _generate_text(self, prompt, temperature, top_p, max_new_tokens, stop=None):
            do_sample = bool(temperature and temperature > 0)
            #print('promp ' , prompt)
            # Tokenize and move to device
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Greedy if temperature==0; sample otherwise
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
                num_beams=1,
            )

            # ONLY decode newly generated tokens
            new_tokens = out[0][input_len:]
            text = self.tok.decode(new_tokens, skip_special_tokens=True)

            # Debug prints (will actually show something now)
            #print("text ,", repr(text))

            # Optional: apply 'stop' only to the NEW text
            if stop:
                for s in stop:
                    if not s:
                        continue
                    i = text.find(s)
                    if i != -1:
                        text = text[:i]
                        break

            return text

        def call_oracle(self, prompt, sched_a,sched_b,temperature=None, top_p=None, max_new_tokens=None, stop=None):
            if temperature is None: temperature = self.default_temperature
            if top_p is None: top_p = self.default_top_p
            if max_new_tokens is None: max_new_tokens = self.default_max_new_tokens
            #print('prompt , ' , prompt)
            txt = self._generate_text(prompt, temperature, top_p, max_new_tokens, stop)
            s = txt.strip()
            #print('s , ' , s)
            if s.endswith("}"):
                for c in ("A","B"):
                    if "{%s}"%c in s:
                        return c, s
            c = s[:1] if s and s[0] in "AB" else "A"
            return c, s

    _CLIENT = _OptimizedClient(tok, model)
    return _CLIENT