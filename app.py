import os
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. RAG KNOWLEDGE BASE
manual_docs = [
    Document(
        page_content="TSB-BMS-304: Error code ERR_BMS_304 indicates a Critical Cell Voltage Imbalance. "
                     "Threshold delta exceeds 0.35V between parallel cell groups. "
                     "Action Protocol: Disconnect main contactor, verify wiring harness terminal B, "
                     "replace battery wiring harness SKU: WH-S1-04, and execute BMS firmware recalibration v2.4.",
        metadata={"category": "BMS", "code": "ERR_BMS_304"}
    ),
    Document(
        page_content="TSB-CAN-102: Error code ERR_CAN_BUS_102 denotes a CAN Bus Communication Timeout. "
                     "Occurs when MCU (Motor Controller Unit) fails heartbeat acknowledgement for >500ms. "
                     "Action Protocol: Inspect 12V auxiliary line, check termination resistor (120 Ohm), "
                     "replace MCU communication bridge SKU: MCU-BRG-01.",
        metadata={"category": "Powertrain", "code": "ERR_CAN_BUS_102"}
    ),
    Document(
        page_content="TSB-THM-501: Error code ERR_THM_OVERHEAT_501 indicates Pack Thermal Runaway Warning. "
                     "Temperature sensors read >62C on central thermal block. "
                     "Action Protocol: Quarantine vehicle for 45 mins in cooling bay. Inspect liquid cooling loops "
                     "and replace Thermal Interface Pad SKU: TIP-MOD-09.",
        metadata={"category": "Thermal", "code": "ERR_THM_501"}
    )
]

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(manual_docs, embedding_fn)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 2. TELEMETRY & INVENTORY DB
MOCK_TELEMETRY = {
    "S1P-MUM-4401": {"model": "Ola S1 Pro Gen 2", "dtc": "ERR_BMS_304", "delta_v": 0.41, "temp": "38C"},
    "S1A-BLR-9022": {"model": "Ola S1 Air", "dtc": "ERR_CAN_BUS_102", "delta_v": 0.05, "temp": "31C"},
    "S1X-DEL-1120": {"model": "Ola S1 X+", "dtc": "ERR_THM_OVERHEAT_501", "delta_v": 0.02, "temp": "64C"}
}

MOCK_INVENTORY = {
    "WH-S1-04": {"name": "BMS Wiring Harness", "stock": 5, "bay": "Rack C-02", "status": "Available"},
    "MCU-BRG-01": {"name": "MCU Communication Bridge", "stock": 0, "bay": "N/A", "status": "Out of Stock - Transit from Pune"},
    "TIP-MOD-09": {"name": "Thermal Interface Pad", "stock": 12, "bay": "Rack A-11", "status": "Available"}
}

def agentic_diagnose(vin_selection, hub_location):
    vehicle = MOCK_TELEMETRY[vin_selection]
    retrieved = retriever.invoke(vehicle["dtc"])
    rag_text = retrieved[0].page_content if retrieved else "No TSB found."

    sku = next((k for k in MOCK_INVENTORY if k in rag_text), None)
    inv = MOCK_INVENTORY.get(sku, {"name": "N/A", "stock": 0, "status": "Unavailable", "bay": "N/A"})

    trace = [
        f"🔍 [Perception] Ingested CAN data from {vin_selection} ({vehicle['model']})",
        f"📡 [Telemetry Scan] DTC: {vehicle['dtc']} | Voltage Delta: {vehicle['delta_v']}V | Temp: {vehicle['temp']}",
        f"📚 [Action: RAG Query] Retrieved official engineering manual for {vehicle['dtc']}",
        f"⚙️ [Action: ERP Check] Checked hub warehouse for SKU '{sku}' at {hub_location}",
        f"📦 [Observation] Status: {inv['status']} ({inv['stock']} in stock)"
    ]

    ticket = f"""### 🛠️ OLA TECHNICIAN WORK ORDER
**Vehicle:** {vehicle['model']} | **Location:** {hub_location}
**Root Cause:** {vehicle['dtc']} (Delta: {vehicle['delta_v']}V)

#### 1. Mandatory Repair Checklist:
* {rag_text}

#### 2. Warehouse Pick-List:
* **Part:** {inv['name']} (`{sku}`)
* **Location:** {inv['bay']}
* **Availability:** {'✅ In Stock (' + str(inv['stock']) + ' units)' if inv['stock'] > 0 else '⚠️ Out of Stock'}

#### 3. Sign-off Criteria:
* Clear active DTC `{vehicle['dtc']}` using MoveOS Diagnostic Flasher.
"""
    return "\n\n".join(trace), rag_text, ticket

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# ⚡ Ola Electric — Field Diagnostic Copilot")
    with gr.Row():
        with gr.Column(scale=1):
            vin = gr.Dropdown(choices=list(MOCK_TELEMETRY.keys()), value="S1P-MUM-4401", label="Vehicle in Bay")
            hub = gr.Dropdown(choices=["Mumbai - Powai Hub", "Bangalore - Koramangala Hub"], value="Mumbai - Powai Hub", label="Service Hub")
            btn = gr.Button("⚡ Run Autonomous Diagnosis", variant="primary")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🚀 Work Order"):
                    out_ticket = gr.Markdown()
                with gr.TabItem("🧠 Agent Reasoning Loop"):
                    out_trace = gr.Textbox(lines=6)
                with gr.TabItem("📖 Grounded TSB"):
                    out_rag = gr.Textbox(lines=4)

    btn.click(agentic_diagnose, inputs=[vin, hub], outputs=[out_trace, out_rag, out_ticket])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
