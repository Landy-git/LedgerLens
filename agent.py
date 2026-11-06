import base64
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "10"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "300"))

# Thinking models (e.g. Qwen3) are much faster with reasoning off; leave empty to omit the parameter
_THINKING_FLAG = os.getenv("VISION_ENABLE_THINKING", "").strip().lower()
VISION_ENABLE_THINKING: bool | None = (
    True if _THINKING_FLAG in {"1", "true", "yes"} else
    False if _THINKING_FLAG in {"0", "false", "no"} else None
)
VISION_JSON_MODE = os.getenv("VISION_JSON_MODE", "").strip().lower() in {"1", "true", "yes"}

# PDF pages with fewer extracted characters are treated as scans and rendered to images
MIN_TEXT_CHARS = 30

SYSTEM_PROMPT = """你是一名严谨的财务票据信息提取助手。请读取用户提供的票据（图片，或 PDF 中抽取的文本），整理成一个 JSON 对象。

规则：
1. 只输出一个 JSON 对象，不要输出 Markdown 代码块、解释或任何其他文字。
2. 严格按下方字段输出；票据上没有的信息填 null，不要猜测或编造。
3. 金额一律输出纯数字（不带千分位逗号、货币符号或单位），单位为元。
4. 日期统一为 YYYY-MM-DD；无法确定完整日期时填 null。
5. category 只能从以下选项中选择：餐饮、交通、住宿、办公、通讯、服务、货物、其他。
6. bill_type 只能从以下选项中选择：增值税专用发票、增值税普通发票、电子发票、收据、行程单、账单、其他。
7. items 为明细行数组（最多 30 行），没有明细则输出 []。每行包含 name、quantity、unit_price、amount、tax_rate。
8. raw_text 为票据内容逐行忠实转写，保留抬头、项目、金额、税号、日期等关键信息；不要添加票据上不存在的内容。
9. 若输入包含多页且属于同一张票据，请合并各页信息后输出。

输出字段（键名固定）：
- bill_type：票据类型
- invoice_code：发票代码
- invoice_number：发票号码
- issue_date：开票日期
- seller_name：销售方/供应商名称
- seller_tax_id：销售方纳税人识别号
- buyer_name：购买方名称
- buyer_tax_id：购买方纳税人识别号
- amount_total：价税合计（含税总额）
- amount_excl_tax：不含税金额
- tax_amount：税额
- currency：币种代码，如 CNY；无法判断填 null
- category：费用类别
- items：明细行数组
- remarks：备注或其他有用信息
- raw_text：票据全文转写"""


def _as_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class TicketItem(BaseModel):
    name: str | None = Field(None, description="项目名称")
    quantity: float | None = Field(None, description="数量")
    unit_price: float | None = Field(None, description="单价")
    amount: float | None = Field(None, description="金额")
    tax_rate: str | None = Field(None, description="税率，如 13%")

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _numbers(cls, value: object) -> float | None:
        return _as_number(value)

    @field_validator("name", "tax_rate", mode="before")
    @classmethod
    def _texts(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class TicketData(BaseModel):
    bill_type: str | None = Field(None, description="票据类型")
    invoice_code: str | None = Field(None, description="发票代码")
    invoice_number: str | None = Field(None, description="发票号码")
    issue_date: str | None = Field(None, description="开票日期 YYYY-MM-DD")
    seller_name: str | None = Field(None, description="销售方名称")
    seller_tax_id: str | None = Field(None, description="销售方纳税人识别号")
    buyer_name: str | None = Field(None, description="购买方名称")
    buyer_tax_id: str | None = Field(None, description="购买方纳税人识别号")
    amount_total: float | None = Field(None, description="价税合计")
    amount_excl_tax: float | None = Field(None, description="不含税金额")
    tax_amount: float | None = Field(None, description="税额")
    currency: str | None = Field(None, description="币种")
    category: str | None = Field(None, description="费用类别")
    items: list[TicketItem] = Field(default_factory=list, description="明细行")
    remarks: str | None = Field(None, description="备注")
    raw_text: str | None = Field(None, description="票据全文转写")

    @field_validator("amount_total", "amount_excl_tax", "tax_amount", mode="before")
    @classmethod
    def _amounts(cls, value: object) -> float | None:
        return _as_number(value)

    @field_validator(
        "bill_type",
        "invoice_code",
        "invoice_number",
        "issue_date",
        "seller_name",
        "seller_tax_id",
        "buyer_name",
        "buyer_tax_id",
        "currency",
        "category",
        "remarks",
        "raw_text",
        mode="before",
    )
    @classmethod
    def _texts(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("items", mode="before")
    @classmethod
    def _items(cls, value: object) -> object:
        return [] if value is None else value


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("VISION_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置视觉模型 API Key。请在项目根目录创建 .env 并设置 VISION_API_KEY（参考 .env.example）。"
        )
    extra: dict = {}
    if VISION_ENABLE_THINKING is not None:
        extra["extra_body"] = {"enable_thinking": VISION_ENABLE_THINKING}
    if VISION_JSON_MODE:
        extra["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(
        model=os.getenv("VISION_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("VISION_BASE_URL", DEFAULT_BASE_URL),
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=VISION_TIMEOUT,
        **extra,
    )


def _image_data_url(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA", "P"}:
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = image.convert("RGB")
            rgb.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=85)
    except Exception as exc:
        raise ValueError("无法读取图片，请确认文件是有效的 PNG/JPG 图片") from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _load_pdf_pages(file_bytes: bytes) -> tuple[list[dict], int]:
    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("无法打开 PDF 文件，文件可能已损坏或加密") from exc
    try:
        if document.needs_pass:
            raise ValueError("PDF 已加密，请先解除密码保护后再上传")
        total = document.page_count
        if total == 0:
            raise ValueError("PDF 中没有页面")

        pages: list[dict] = []
        for index in range(min(total, MAX_PDF_PAGES)):
            page = document.load_page(index)
            text = page.get_text("text").strip()
            if len(re.sub(r"\s+", "", text)) >= MIN_TEXT_CHARS:
                pages.append({"page": index + 1, "type": "text", "text": text})
            else:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pages.append(
                    {"page": index + 1, "type": "image", "data_url": _image_data_url(pixmap.tobytes("png"))}
                )
        return pages, total
    finally:
        document.close()


def _load_pages(file_bytes: bytes, filename: str) -> tuple[list[dict], int]:
    suffix = Path(filename).suffix.lower()
    if not file_bytes:
        raise ValueError("文件内容为空")
    if suffix == ".pdf":
        return _load_pdf_pages(file_bytes)
    if suffix in IMAGE_SUFFIXES:
        return [{"page": 1, "type": "image", "data_url": _image_data_url(file_bytes)}], 1
    raise ValueError(f"不支持的文件类型：{suffix or '未知'}，请上传 PNG/JPG/PDF")


def _build_messages(pages: list[dict], filename: str) -> list:
    content: list[dict] = [
        {"type": "text", "text": f"文件名：{filename}。请识别以下票据内容，并按系统要求输出 JSON。"}
    ]
    for page in pages:
        if page["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": page["data_url"]}})
        else:
            content.append({"type": "text", "text": f"【第 {page['page']} 页文本】\n{page['text']}"})
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _parse_model_json(text: str) -> dict:
    cleaned = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出中没有找到 JSON 对象")
    return json.loads(cleaned[start : end + 1])


def extract_ticket(file_bytes: bytes, filename: str) -> dict:
    """Send a ticket image/PDF to the vision model and return validated structured JSON."""
    pages, page_count = _load_pages(file_bytes, filename)
    llm = _get_llm()
    messages = _build_messages(pages, filename)

    last_error: Exception | None = None
    last_output = ""
    last_finish_reason: str | None = None
    for _ in range(2):
        try:
            reply = llm.invoke(messages)
        except APITimeoutError as exc:
            raise RuntimeError(
                f"调用视觉模型超时（{VISION_TIMEOUT} 秒）。可在 .env 中调高 VISION_TIMEOUT，"
                "或换用更快的模型（如 qwen3.8-flash）。"
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError("无法连接视觉模型服务，请检查网络和 VISION_BASE_URL 配置。") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"视觉模型接口返回错误（HTTP {exc.status_code}）：{exc.message}") from exc
        text = _message_text(reply)
        finish_reason = reply.response_metadata.get("finish_reason")
        last_output, last_finish_reason = text, finish_reason
        try:
            bill = TicketData.model_validate(_parse_model_json(text))
        except (ValueError, ValidationError) as exc:
            last_error = exc
            if finish_reason == "length":
                raise ValueError(
                    f"模型输出被截断：MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS} 被思考过程或正文耗尽，"
                    "没有拿到完整的 JSON。请在 .env 中调高 MAX_OUTPUT_TOKENS（例如 8192）后重试。"
                ) from exc
            reason = str(exc).splitlines()[0][:300]
            messages = messages + [
                AIMessage(content=text),
                HumanMessage(
                    content=f"上面的输出无法作为合法的票据 JSON 使用：{reason}。"
                    "请只输出一个 JSON 对象，键名与字段要求一致，不要代码块或解释。"
                ),
            ]
            continue

        result = bill.model_dump()
        result["source"] = {
            "filename": filename,
            "page_count": page_count,
            "pages_processed": len(pages),
            "truncated": page_count > len(pages),
        }
        return result

    raise ValueError(
        f"模型输出无法解析为合法 JSON：{last_error}"
        f"（finish_reason={last_finish_reason}，输出预览：{last_output[:200]!r}）"
    )


def render_pdf_preview(file_bytes: bytes) -> bytes | None:
    """Render the first PDF page as a PNG for UI previews."""
    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        if document.needs_pass or document.page_count == 0:
            return None
        pixmap = document.load_page(0).get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()
