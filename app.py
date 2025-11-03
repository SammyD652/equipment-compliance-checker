# app.py
import os, io, json, math, tempfile
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from openai import OpenAI

st.set_page_config(page_title="Equipment ↔ Submittal Compliance Checker", layout="wide")
st.title("Equipment ↔ Submittal Compliance Checker")

# ---------- Configuration ----------
AGENT_INSTRUCTIONS = """
You are the Equipment ↔ Submittal Compliance Agent. Your job is to:
- Ingest submittal PDFs, nameplate images, and CSV/XLSX tables.
- Extract structured equipment records.
- Normalize units/labels.
- Match submittal records to nameplate records.
- Compare key parameters using tolerance rules.
- Produce an Excel report and a concise summary of mismatches.

STRICT JSON ONLY. Use this schema (null if unknown):
[{
  "source_id": null | string,
  "equipment_id": null | string,
  "tag": null | string,
  "model": null | string,
  "manufacturer": null | string,
  "serial": null | string,
  "rated_power_kw": null | number,
  "voltage_v": null | number,
  "frequency_hz": null | number,
  "phase": null | string,
  "flow_l_s": null | number,
  "pressure_kpa": null | number,
  "temperature_c": null | number,
  "notes": null | string
}]
Return ONLY valid JSON.
"""

PAGES_PER_CHUNK = 8   # reduce to 4 if needed

# ---------- Helpers ----------
def get_openai_client():
    key = (
        st.secrets.get("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or st.session_state.get("OPENAI_API_KEY")
    )
    if not key:
        st.error("No OpenAI API key found. In Streamlit: ••• → Settings → Secrets → `OPENAI_API_KEY = \"sk-...\"`")
        st.stop()
    st.session_state["OPENAI_API_KEY"] = key
    return OpenAI(api_key=key)

def pdf_pages_as_images(pdf_bytes, dpi=200, start=0, end=None):
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=(start+1), last_page=(end if end else None))
    bufs = []
    for im in pages:
        b = io.BytesIO()
        im.save(b, format="PNG")
        bufs.append(b.getvalue())
    return bufs

def parse_json(loose_text):
    # Try to coerce to valid JSON array even if model added extra text.
    try:
        return json.loads(loose_text)
    except Exception:
        s = loose_text.find("["); e = loose_text.rfind("]")
        if s != -1 and e != -1 and e > s:
            return json.loads(loose_text[s:e+1])
        raise

def call_openai_vision(client, images_bytes_list, instruction):
    content = [{"type":"text","text": instruction}]
    for b in images_bytes_list:
        content.append({"type":"input_image","image_bytes": b})
    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role":"user","content":content}],
        temperature=0
    )
    return parse_json(resp.output_text)

def call_openai_text(client, text_chunk, instruction):
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role":"system","content": instruction},
            {"role":"user","content": f"Extract per schema from this text:\n\n{text_chunk}"}
        ],
        temperature=0
    )
    return parse_json(resp.output_text)

# ---------- UI ----------
pdf = st.file_uploader("Upload technical submittal PDF (large allowed)", type=["pdf"])
nameplate_imgs = st.file_uploader("Upload nameplate images (optional)", type=["png","jpg","jpeg"], accept_multiple_files=True)

# Optional manual key (useful for local dev)
api_key_input = st.text_input("🔐 OpenAI API Key (leave blank if set in Secrets)", type="password")
if api_key_input:
    st.session_state["OPENAI_API_KEY"] = api_key_input

process = st.button("▶️ Process files", type="primary")

# ---------- Pipeline ----------
if process:
    if not pdf:
        st.warning("Please upload a technical submittal PDF first.")
        st.stop()

    client = get_openai_client()
    pdf_bytes = pdf.read()

    # Fast text scrape
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
    num_pages = len(reader.pages)
    chunks = max(1, math.ceil(num_pages / PAGES_PER_CHUNK))

    st.write(f"Pages: {num_pages} → chunks: {chunks}")
    progress = st.progress(0)
    status = st.empty()

    all_records = []

    # Text path (cheap & quick)
    try:
        if all_text.strip():
            status.info("Extracting from embedded text…")
            txt_records = call_openai_text(client, all_text[:100_000], AGENT_INSTRUCTIONS)
            all_records.extend(txt_records)
    except Exception as e:
        st.warning(f"Text extraction skipped: {e}")

    # Vision OCR path
    for c in range(chunks):
        start = c * PAGES_PER_CHUNK
        end = min(num_pages, (c+1)*PAGES_PER_CHUNK)
        status.info(f"OCR pages {start+1}–{end}…")
        try:
            imgs = pdf_pages_as_images(pdf_bytes, dpi=200, start=start, end=end)
            if imgs:
                vis_records = call_openai_vision(client, imgs, AGENT_INSTRUCTIONS)
                all_records.extend(vis_records)
        except Exception as e:
            st.warning(f"OCR failed on pages {start+1}–{end}: {e}")
        progress.progress(int((c+1)/chunks*100))

    # Nameplate images (optional)
    if nameplate_imgs:
        status.info("Parsing nameplate images…")
        try:
            nb = [f.read() for f in nameplate_imgs]
            vis_records = call_openai_vision(client, nb, AGENT_INSTRUCTIONS)
            all_records.extend(vis_records)
        except Exception as e:
            st.warning(f"Nameplate parse failed: {e}")

    status.empty()

    # Normalize
    if not all_records:
        st.error("No records extracted. Try a smaller page chunk size or a clearer PDF.")
        st.stop()

    df = pd.DataFrame(all_records).drop_duplicates()

    # Placeholder matching logic (customize later)
    def norm(x): return (str(x).strip().lower() if x is not None else "")
    df["__key"] = (df.get("tag","").apply(norm) + "|" + df.get("model","").apply(norm))

    # Build Excel
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        xls_path = tmp.name
    with pd.ExcelWriter(xls_path, engine="xlsxwriter") as xl:
        df.to_excel(xl, index=False, sheet_name="Raw_Extract")
        pd.DataFrame([], columns=["todo"]).to_excel(xl, index=False, sheet_name="Matches")
        pd.DataFrame([], columns=["todo"]).to_excel(xl, index=False, sheet_name="Mismatches")
        pd.DataFrame([{"items_total": len(df)}]).to_excel(xl, index=False, sheet_name="Summary")

    st.success("Done. Download your report:")
    with open(xls_path, "rb") as f:
        st.download_button("Download REPORT.xlsx", f, file_name="REPORT.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
