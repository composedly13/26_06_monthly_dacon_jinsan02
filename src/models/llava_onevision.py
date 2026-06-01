"""
LLaVA-OneVision vLLM 추론 모듈.
모델 초기화를 한 번만 수행하고 배치 추론을 지원.
"""
from dataclasses import asdict

import torch
from vllm import LLM, SamplingParams
from vllm import EngineArgs
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from typing import Literal


class ReasonAnswer(BaseModel):
    reason: str
    answer_id: Literal["0", "1", "2"]


_JSON_SCHEMA = ReasonAnswer.model_json_schema()
_GUIDED = GuidedDecodingParams(json=_JSON_SCHEMA)


def build_engine_args(model_id: str, max_model_len: int = 16384, gpu_util: float = 0.9) -> EngineArgs:
    return EngineArgs(
        model=model_id,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=gpu_util,
        disable_mm_preprocessor_cache=True,
    )


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>user <image>\n{question}<|im_end|> "
        "<|im_start|>assistant\n"
    )


def init_llm(model_id: str, seed: int = 42, **kwargs) -> LLM:
    engine_args = build_engine_args(model_id, **kwargs)
    args = asdict(engine_args) | {"seed": seed}
    args["limit_mm_per_prompt"] = {"image": 1, "video": 0, "audio": 0}
    return LLM(**args)


def make_sampling_params(temperature: float = 0.0, max_tokens: int = 256) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        guided_decoding=_GUIDED,
    )
