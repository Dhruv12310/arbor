"""
Provider for locally fine-tuned Arbor models.
Requires: pip install transformers bitsandbytes accelerate
"""

from __future__ import annotations

from typing import Optional

from arbor.providers.base import LLMProvider


class ArborFineTunedProvider(LLMProvider):
    """
    Runs a locally fine-tuned model (or any HuggingFace causal LM) with
    4-bit quantization via bitsandbytes for low VRAM usage.

    Args:
        model_path: Local path (e.g. "models/treegen-merged") or HF model ID
                    (e.g. "arbor-ai/treegen-7b").
        max_new_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        except ImportError:
            raise ImportError(
                "transformers and bitsandbytes are required for ArborFineTunedProvider.\n"
                "Install with: pip install transformers bitsandbytes accelerate"
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._temperature = temperature

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._tokenizer = tokenizer
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )

    @property
    def name(self) -> str:
        return f"finetuned/{self.model_path.split('/')[-1]}"

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        content, _ = await self.complete_with_finish_reason(prompt, temperature, max_tokens, chat_history)
        return content

    async def complete_with_finish_reason(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, str]:
        messages = list(chat_history or []) + [{"role": "user", "content": prompt}]
        gen_kwargs: dict = {"do_sample": False}
        if temperature and temperature > 0.0:
            gen_kwargs = {"do_sample": True, "temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens

        result = self._pipeline(messages, **gen_kwargs)
        # pipeline returns the full conversation; extract only the new assistant turn
        output = result[0]["generated_text"]
        if isinstance(output, list):
            assistant_msg = output[-1]
            text = assistant_msg.get("content", "") if isinstance(assistant_msg, dict) else str(assistant_msg)
        else:
            text = str(output)

        return text, "stop"
