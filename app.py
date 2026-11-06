import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

from agent import extract_ticket, render_pdf_preview

# Number of concurrent extraction tasks (1-8); lower it if the gateway rate-limits you
MAX_CONCURRENCY = max(1, min(int(os.getenv("BATCH_CONCURRENCY", "3")), 8))


def _extract_one(name: str, data: bytes) -> dict:
    item: dict = {"filename": name}
    started = time.time()
    try:
        item["result"] = extract_ticket(data, name)
    except Exception as exc:
        item["error"] = str(exc)
    item["elapsed"] = time.time() - started
    return item

st.set_page_config(page_title="LedgerLens · 票据识别", page_icon="🧾", layout="wide")

st.title("🧾 LedgerLens")
st.caption("Read any receipt. Fill the ledger. 一次可上传多个发票、收据或账单的图片 / PDF，逐个提取为结构化 JSON。")

with st.sidebar:
    st.header("Impact")
    st.markdown(
        "**Manual processing:** ~2–3 minutes per bill  \n"
        "*(manual entry + verification)*\n\n"
        "**AI Agent processing:** ~5–10 seconds per bill  \n"
        "*(upload + parsing)*\n\n"
        "**Efficiency gain:** ~70%  \n"
        "*(based on actual project data)*"
    )

uploaded_files = st.file_uploader(
    "上传票据文件（可多选）",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("支持 PNG、JPG、WebP、PDF，可一次选择多个文件，单个文件建议不超过 10 MB。")
    st.stop()

files = [(item.name, item.getvalue()) for item in uploaded_files]

# Clear previous results when the file selection changes. Results live in
# session_state so they survive the rerun triggered by the download button.
digest = hashlib.sha1()
for name, data in files:
    digest.update(name.encode("utf-8"))
    digest.update(hashlib.sha1(data).digest())
digest = digest.hexdigest()
if st.session_state.get("digest") != digest:
    st.session_state.pop("results", None)
    st.session_state.pop("total_elapsed", None)
    st.session_state["digest"] = digest

previewed = files[:9]
with st.expander(f"文件预览（共 {len(files)} 个文件）", expanded=len(files) == 1):
    columns = st.columns(min(len(previewed), 3))
    for index, (name, data) in enumerate(previewed):
        with columns[index % 3]:
            st.markdown(f"**{name}**")
            if name.lower().endswith(".pdf"):
                preview = render_pdf_preview(data)
                if preview:
                    st.image(preview, caption="第 1 页", width="stretch")
                else:
                    st.caption("（无法生成预览）")
            else:
                st.image(data, width="stretch")
            st.caption(f"{len(data) / 1024:.1f} KB")
    if len(files) > len(previewed):
        st.caption(f"仅预览前 {len(previewed)} 个文件，其余 {len(files) - len(previewed)} 个照常识别。")

start_button = st.button(f"开始识别（{len(files)} 个文件）", type="primary", width="stretch")

if start_button:
    results: list[dict | None] = [None] * len(files)
    progress = st.progress(0.0, text="准备识别…")
    total_started = time.time()
    workers = min(MAX_CONCURRENCY, len(files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_one, name, data): index
            for index, (name, data) in enumerate(files)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {"filename": files[index][0], "error": str(exc), "elapsed": 0.0}
            results[index] = item
            progress.progress(
                done / len(files), text=f"已完成 {done}/{len(files)}：{item['filename']}"
            )
    progress.empty()
    st.session_state["results"] = results
    st.session_state["total_elapsed"] = time.time() - total_started

results = st.session_state.get("results")
if results is None:
    st.stop()

succeeded = [item for item in results if "error" not in item]
failed = [item for item in results if "error" in item]
st.success(
    f"识别完成：成功 {len(succeeded)} 个，失败 {len(failed)} 个，"
    f"总耗时 {st.session_state.get('total_elapsed', 0.0):.2f} 秒。"
)

if failed:
    with st.expander(f"识别失败的文件（{len(failed)} 个）", expanded=True):
        for item in failed:
            st.error(f"{item['filename']}：{item['error']}")

if len(succeeded) > 1:
    st.download_button(
        f"下载全部结果（{len(succeeded)} 个 JSON）",
        data=json.dumps([item["result"] for item in succeeded], ensure_ascii=False, indent=2),
        file_name="tickets.json",
        mime="application/json",
        key="download-all",
        width="stretch",
    )

if succeeded:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for item in succeeded:
        stem = Path(item["filename"]).stem
        seen[stem] = seen.get(stem, 0) + 1
        labels.append(stem if seen[stem] == 1 else f"{stem} ({seen[stem]})")

    tabs = st.tabs(labels)
    for index, (tab, item) in enumerate(zip(tabs, succeeded)):
        with tab:
            st.caption(f"{item['filename']} · 识别耗时 {item['elapsed']:.2f} 秒")
            st.json(item["result"])
            st.download_button(
                "下载 JSON",
                data=json.dumps(item["result"], ensure_ascii=False, indent=2),
                file_name=f"{Path(item['filename']).stem}.json",
                mime="application/json",
                key=f"download-{index}",
            )
            if item["result"].get("raw_text"):
                with st.expander("票据原文转写"):
                    st.text(item["result"]["raw_text"])
