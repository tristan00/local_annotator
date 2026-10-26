GREEDY = {"temperature": 0.0}

MODELS = {
    "granite4.2-3b-fp8": {"repository": "ibm-granite/granite-4.2-3b-fp8", "serve": [], "sampling": GREEDY},
    "granite4.2-8b-fp8": {"repository": "ibm-granite/granite-4.2-8b-fp8", "serve": [], "sampling": GREEDY},
    "ministral3-3b": {"repository": "mistralai/Ministral-3-3B-Instruct-2512", "serve": ["--language-model-only"], "sampling": GREEDY},
    "ministral3-8b": {"repository": "mistralai/Ministral-3-8B-Instruct-2512", "serve": ["--language-model-only"], "sampling": GREEDY},
    "gemma4-e4b": {"repository": "google/gemma-4-E4B-it", "serve": ["--language-model-only"], "sampling": GREEDY},
    "gemma4-26b-a4b-nvfp4": {"repository": "nvidia/Gemma-4-26B-A4B-NVFP4", "serve": ["--language-model-only"], "sampling": GREEDY},
    "qwen3-4b-2507-fp8": {"repository": "Qwen/Qwen3-4B-Instruct-2507-FP8", "serve": [], "sampling": GREEDY},
    "qwen3-8b-fp8": {"repository": "Qwen/Qwen3-8B-FP8", "serve": [], "sampling": GREEDY},
    "phi4-mini": {"repository": "microsoft/Phi-4-mini-instruct", "serve": [], "sampling": GREEDY},
    "olmo3-7b": {"repository": "allenai/Olmo-3-7B-Instruct", "serve": [], "sampling": GREEDY},
}
