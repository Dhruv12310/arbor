"""
Provider for locally fine-tuned Arbor models.
Loads HuggingFace models directly via transformers — no HTTP server required.

Requires: pip install transformers bitsandbytes accelerate
"""

from __future__ import annotations

import asyncio
from typing import Optional

from arbor.providers.base import LLMProvider

DEFAULT_MODEL = "TStark12310/arbor-treegen-7b-v2"


class ArborFineTunedProvider(LLMProvider):
    """
    Loads a fine-tuned HuggingFace model directly using transformers.
    Uses 4-bit quantization (bitsandbytes) for low VRAM usage.
    No HTTP server required — runs fully local.

    Args:
        model_path: HF model ID (e.g. "TStark12310/arbor-treegen-7b-v2")
                    or local path (e.g. "models/treegen-merged").
                    Defaults to TStark12310/arbor-treegen-7b-v2.
        max_new_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature (0.0 = greedy).
        max_seq_length: Maximum input token length before truncation.
        load_in_4bit: Use 4-bit quantization (saves VRAM). Default True.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers and bitsandbytes are required for ArborFineTunedProvider.\n"
                "Install with: pip install transformers bitsandbytes accelerate"
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._temperature = temperature
        self.max_seq_length = max_seq_length

        print(f"[ArborFineTunedProvider] Loading {model_path} ...")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        print(f"[ArborFineTunedProvider] Ready.")

    @property
    def name(self) -> str:
        return f"finetuned/{self.model_path.split('/')[-1]}"

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _build_messages(
        self,
        prompt: str,
        chat_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Build message list from chat_history + new user prompt."""
        messages = list(chat_history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_sync(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int],
    ) -> tuple[str, str]:
        """Synchronous generation using model.generate() directly."""
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]
        max_new = max_tokens or self.max_new_tokens

        gen_kwargs: dict = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "repetition_penalty": 1.1,
        }
        if temperature and temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Detect truncation
        finish = "length" if len(new_tokens) >= max_new else "stop"
        return text, finish

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        content, _ = await self.complete_with_finish_reason(
            prompt, temperature, max_tokens, chat_history
        )
        return content

    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        messages = self._build_messages(prompt, chat_history)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            messages,
            temperature,
            max_tokens,
        )
