"""
프롬프트 템플릿 모음.
각 함수는 (context, question, answers) → prompt_text 반환.
"""


def build_baseline_prompt(context: str, question: str, answers: list[str]) -> str:
    pre = (
        "You are an expert Vision Language assistant. "
        "When given an image, a context, a question, and options, "
        "you MUST respond only with a JSON object"
    )
    post = (
        "Give the output in strict JSON format: \n"
        "{\n"
        '   "reason": "One short sentence of reasoning.",\n'
        '   "answer_id": "<one of: 0, 1, 2>"\n'
        "}\n"
    )
    ctx = f"Context: {context}"
    q = f"Question: {question}"
    opts = "Options:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(answers))

    return "\n".join([pre, ctx, q, opts, post, "Do NOT output multiple options."])


def build_cot_prompt(context: str, question: str, answers: list[str]) -> str:
    """Chain-of-thought 프롬프트 — 더 긴 추론을 유도."""
    ctx = f"Context: {context}"
    q = f"Question: {question}"
    opts = "Options:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(answers))

    instruction = (
        "You are a careful, unbiased Vision Language assistant. "
        "Analyze the image and text carefully. "
        "Consider whether the question can be answered from the given evidence alone. "
        "If there is insufficient evidence, choose the 'cannot be determined' option. "
        "Respond ONLY with a JSON object."
    )
    post = (
        "Output strict JSON:\n"
        "{\n"
        '  "reason": "2-3 sentence step-by-step reasoning.",\n'
        '  "answer_id": "<0, 1, or 2>"\n'
        "}"
    )
    return "\n".join([instruction, ctx, q, opts, post])
