GREEDY = {"temperature": 0.0}
LOW_REASONING = {"temperature": 0.0, "reasoning_effort": "low"}
GATED_DELTA = ["--language-model-only", "--prefix-match-unit", "32"]

MODELS = {
    "granite4.2-3b-fp8": {"repository": "ibm-granite/granite-4.2-3b-fp8", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "granite4.2-8b-fp8": {"repository": "ibm-granite/granite-4.2-8b-fp8", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "ministral3-3b": {"repository": "mistralai/Ministral-3-3B-Instruct-2512", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "ministral3-8b": {"repository": "mistralai/Ministral-3-8B-Instruct-2512", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "gemma4-e4b": {"repository": "google/gemma-4-E4B-it", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "gemma4-26b-a4b-nvfp4": {"repository": "nvidia/Gemma-4-26B-A4B-NVFP4", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "qwen3-4b-2507-fp8": {"repository": "Qwen/Qwen3-4B-Instruct-2507-FP8", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "qwen3-8b-fp8": {"repository": "Qwen/Qwen3-8B-FP8", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "phi4-mini": {"repository": "microsoft/Phi-4-mini-instruct", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "olmo3-7b": {"repository": "allenai/Olmo-3-7B-Instruct", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "qwen3-14b-fp8": {"repository": "Qwen/Qwen3-14B-FP8", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "qwen3-30b-a3b-2507-awq": {"repository": "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "qwen3-32b-nvfp4": {"repository": "RedHatAI/Qwen3-32B-NVFP4", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "glm4.7-flash-awq": {"repository": "QuantTrio/GLM-4.7-Flash-AWQ", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "mistral-small3.2-24b-nvfp4": {"repository": "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "ministral3-14b": {"repository": "mistralai/Ministral-3-14B-Instruct-2512", "serve": ["--language-model-only"], "sampling": GREEDY, "max_tokens": 160},
    "qwen3.6-35b-a3b-nvfp4": {"repository": "nvidia/Qwen3.6-35B-A3B-NVFP4", "serve": [*GATED_DELTA, "--max-num-seqs", "64"], "sampling": GREEDY, "max_tokens": 160},
    "qwen3.8-27b-nvfp4": {"repository": "nvidia/Qwen3.8-27B-NVFP4", "serve": GATED_DELTA, "sampling": GREEDY, "max_tokens": 160},
    "nemotron3.5-lightning-30b-a3b-nvfp4": {"repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "serve": [], "sampling": GREEDY, "max_tokens": 160},
    "qwen3.5-4b": {"repository": "Qwen/Qwen3.5-4B", "serve": GATED_DELTA, "sampling": GREEDY, "max_tokens": 160},
    "gpt-oss-20b": {"repository": "openai/gpt-oss-20b", "serve": ["--reasoning-parser", "openai_gptoss"], "sampling": LOW_REASONING,
                    "max_tokens": 1024},
}
