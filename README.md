# LedgerLens

**Read any receipt. Fill the ledger.**

An AI agent that turns invoice, receipt and bill images or PDFs into structured JSON. It combines a vision LLM (Qwen-VL via the DashScope OpenAI-compatible API) with LangChain and a Streamlit UI.

## Features

- Upload multiple PNG / JPG / WebP / PDF files at once
- PDF handling: extract the text layer when present, rasterize scanned pages, detect encrypted files
- Image preprocessing: EXIF orientation correction, transparent-background flattening, downscaling to limit vision tokens
- Strict JSON extraction validated with Pydantic, with one automatic retry when the model output is malformed
- Optional reasoning-off mode and JSON mode for lower latency and higher reliability
- Concurrent batch processing (configurable, default 3 workers)
- Per-file preview, JSON view, original-text transcription, individual and combined JSON downloads
- Failed files are reported individually and do not block the rest of the batch

## How it works

1. **Load** - PDF pages with a text layer are passed as text; scanned pages are rendered to images. Uploaded images are normalized and compressed.
2. **Extract** - The pages are sent to a vision model with a fixed field schema in the prompt.
3. **Validate** - The response is parsed as JSON and validated against the `TicketData` model. On invalid output the model is asked once more with the validation error.
4. **Return** - The validated fields, the full transcribed text (`raw_text`) and source metadata are returned.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set VISION_API_KEY
streamlit run app.py
```

Tested with Python 3.13 on Windows.

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `VISION_API_KEY` | - | API key for the vision model (required) |
| `VISION_BASE_URL` | DashScope compatible endpoint | OpenAI-compatible base URL |
| `VISION_MODEL` | `qwen-vl-max` | Vision model name, e.g. `qwen3.8-flash` |
| `VISION_ENABLE_THINKING` | empty | `false` disables reasoning on thinking models; empty omits the parameter |
| `VISION_JSON_MODE` | `false` | Force `response_format={"type": "json_object"}` |
| `VISION_TIMEOUT` | `300` | Request timeout in seconds |
| `MAX_OUTPUT_TOKENS` | `8192` | Max output tokens per request; thinking tokens count toward this |
| `MAX_PDF_PAGES` | `10` | Maximum PDF pages processed per file |
| `MAX_IMAGE_SIDE` | `2000` | Longest image side after downscaling, in pixels |
| `BATCH_CONCURRENCY` | `3` | Concurrent extraction tasks (1-8) |

The app expects a multimodal model. Text-only models such as `deepseek-chat` cannot read images.

## Output

Each file returns one JSON object:

```json
{
  "bill_type": "增值税普通发票",
  "invoice_code": "031002100311",
  "invoice_number": "12345678",
  "issue_date": "2026-10-29",
  "seller_name": "Example Catering Co., Ltd.",
  "seller_tax_id": null,
  "buyer_name": "Example Fund Management Co., Ltd.",
  "buyer_tax_id": null,
  "amount_total": 106.0,
  "amount_excl_tax": 100.0,
  "tax_amount": 6.0,
  "currency": "CNY",
  "category": "餐饮",
  "items": [
    {
      "name": "Business meal",
      "quantity": 1.0,
      "unit_price": 100.0,
      "amount": 100.0,
      "tax_rate": "6%"
    }
  ],
  "remarks": null,
  "raw_text": "...",
  "source": {
    "filename": "invoice.png",
    "page_count": 1,
    "pages_processed": 1,
    "truncated": false
  }
}
```

Missing fields are `null`; `items` is always an array. `source.truncated` is `true` when the PDF has more pages than `MAX_PDF_PAGES`.

## Project layout

```
.
├── app.py            # Streamlit UI, multi-file upload, concurrent batch processing
├── agent.py          # PDF/image preprocessing, vision model call, schema validation
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- Amounts and dates come from a model and should be verified before any financial use.
- Typical latency is roughly 3-5 seconds per file with reasoning disabled; batch time is reduced by `BATCH_CONCURRENCY` and bounded by the provider's rate limits.
