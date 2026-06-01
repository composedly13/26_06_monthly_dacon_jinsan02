import json
import base64
from pathlib import Path
from io import BytesIO

from PIL import Image


def load_image(image_path: str | Path, img_size: int = 512, as_base64: bool = False):
    try:
        img = Image.open(str(image_path)).convert("RGB")
        w_pct = img_size / float(img.size[0])
        new_h = int(img.size[1] * w_pct)
        img = img.resize((img_size, new_h), Image.LANCZOS)

        if as_base64:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        return img
    except Exception as e:
        print(f"[load_image] {image_path}: {e}")
        return None


def parse_answers(raw: str) -> list[str]:
    return json.loads(raw)


def normalize_label(value) -> str:
    if value is None:
        return "0"
    s = str(value).strip()
    return s if s in {"0", "1", "2"} else "0"


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return json.loads(text)
