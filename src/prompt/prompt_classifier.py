import json
import re
from typing import Optional

from src.llm.shared_pipeline import load_shared_generation_pipeline
from src.rag.settings import load_rag_settings


def load_llm():
    return load_shared_generation_pipeline()


def _parse_label_from_output(output: str, prefer_json: bool) -> int:
    output = (output or "").strip()
    if prefer_json:
        brace = re.search(r"\{[^{}]*\}", output)
        if brace:
            try:
                obj = json.loads(brace.group())
                if isinstance(obj, dict) and "label" in obj:
                    v = int(obj["label"])
                    if v in (0, 1, 2, 3):
                        return v
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    match = re.search(r"(?<![0-9])([0-3])(?![0-9])", output)
    if match:
        return int(match.group(1))
    match2 = re.search(r"[0-3]", output)
    if match2:
        return int(match2.group())
    return -1


def classify_news(text, llm, prefer_json: Optional[bool] = None):
    if prefer_json is None:
        prefer_json = bool(load_rag_settings().get("prefer_json_output", True))

    if prefer_json:
        prompt = f"""
You are a strict AG News classifier.
Choose exactly one label id:
0 = World (politics, government, international affairs, wars)
1 = Sports (games, teams, athletes, tournaments)
2 = Business (companies, finance, markets, economy, deals)
3 = Sci/Tech (technology, science, software, gadgets, research)

Examples:
Text: The UN Security Council approved a new resolution on Middle East peace talks.
Label: 0
Text: Manchester United won 3-1 in the Premier League opener.
Label: 1
Text: Apple reported quarterly revenue growth driven by iPhone sales.
Label: 2
Text: Researchers released a new open-source large language model for coding.
Label: 3

Now classify:
Text: {text}

Output format (strict):
- Return ONE JSON object only, no markdown fences, no extra text.
- Schema: {{"label": <int>}} where <int> is 0, 1, 2, or 3.

Answer:
"""
        max_tokens = 32
    else:
        prompt = f"""
You are a strict AG News classifier.
Choose exactly one label id:
0 = World (politics, government, international affairs, wars)
1 = Sports (games, teams, athletes, tournaments)
2 = Business (companies, finance, markets, economy, deals)
3 = Sci/Tech (technology, science, software, gadgets, research)

Examples:
Text: The UN Security Council approved a new resolution on Middle East peace talks.
Label: 0
Text: Manchester United won 3-1 in the Premier League opener.
Label: 1
Text: Apple reported quarterly revenue growth driven by iPhone sales.
Label: 2
Text: Researchers released a new open-source large language model for coding.
Label: 3

Now classify:
Text: {text}

Output format rules:
- Output one and only one character: 0 or 1 or 2 or 3
- Do not output words, punctuation, or explanation
Answer:
"""
        max_tokens = 3

    result = llm(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_full_text=False,
    )

    output = result[0]["generated_text"].strip()
    return _parse_label_from_output(output, prefer_json)
