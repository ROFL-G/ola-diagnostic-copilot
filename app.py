import os
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. EXPANDED DOMAIN TECHNICAL MANUAL (TSB / RAG KNOWLEDGE BASE)
TSB_DATABASE = [
    {
        "code": "ERR_BMS_304",
        "category": "BMS & High Voltage",
        "content": "TSB-BMS-304: Critical Cell Voltage Imbalance. Threshold delta exceeds 0.35V between parallel cell groups. Action: Isolate main HV contactor, check 16-pin harness continuity at terminal B, replace Battery Wiring Harness SKU: WH-S1-04, and flash BMS firmware calibration v2.4 via MoveOS Service Tool."
    },
    {
        "code": "ERR_CAN_BUS_102",
        "category": "Powertrain & CAN",
        "content": "TSB-CAN-102: CAN Bus Communication Loss / MCU Timeout. Occurs when Motor Controller Unit fails heartbeat acknowledgement for >500ms. Action: Inspect 12V auxiliary line voltage, measure termination resistance (spec: 60 Ohm bus total / 120 Ohm node), replace MCU Communication Bridge SKU: MCU-BRG-01."
    },
    {
        "code": "ERR_THM_OVERHEAT_501",
        "category": "Thermal Systems",
        "content": "TSB-THM-501: Pack Thermal Runaway Warning. Thermistors report cell block temperature >62C. Action: Move scooter to dedicated quarantine cooling bay for 45 mins. Verify radiator fin integrity, inspect pump flow, and install High-Conductivity Thermal Interface Pad SKU: TIP-MOD-09."
    },
    {
        "code": "ERR_CHG_LOCK_208",
        "category": "Charging Subsystem",
        "content": "TSB-CHG-208: Charge Port Solenoid Lock Failure. MoveOS detects solenoid actuator failure to engage latch pin during DC Fast Charge / Home Charging. Action: Clean debris from charge inlet socket, inspect latch pin travel, replace Charge Port Lock Actuator SKU: CPL-ACT-12."
    },
    {
        "code": "ERR_VCPU_OS_771",
        "category": "Instrument Cluster & Telematics",
        "content": "TSB-VCPU-771: 7-inch TFT Dash Screen Blanking / MoveOS Bootloop. VCPU boot sequence timeout caused by corrupted partition during OTA background installation. Action: Connect physical MoveOS Debug Harness, perform cold hardware reset, replace VCPU Central Telematics Unit SKU: VCPU-GEN2-03."
    }
]

# Vector index initialization (<15MB RAM footprint)
docs = [item["content"] for item in TSB_DATABASE]
vectorizer = TfidfVectorizer().fit(docs)
doc_vectors = vectorizer.transform(docs)

def retrieve_tsb(query_code: str) -> str:
    query_vec = vectorizer.transform([query_code])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    return TSB_DATABASE[similarities.argmax()]["content"]

# 2. REALISTIC OLA VEHICLES CURRENTLY IN SERVICE BAY
MOCK_TELEMETRY = {
    "S1P-MUM-4401 (MH-02-EQ-4401)": {
        "model": "Ola S1 Pro Gen 2",
        "dtc": "ERR_BMS_304",
        "delta_v": 0.42,
        "temp": "38°C",
        "soc": "14%",
        "odometer": "18,420 km"
    },
    "S1A-BLR-9022 (KA-03-JJ-9022)": {
        "model": "Ola S1 Air",
        "dtc": "ERR_CAN_BUS_102",
        "delta_v": 0.04,
        "temp": "31°C",
        "soc": "62%",
        "odometer": "9,150 km"
    },
    "S1X-DEL-1120 (DL-01-AB-1120)": {
        "model": "Ola S1 X+ (3kWh)",
        "dtc": "ERR_THM_OVERHEAT_501",
        "delta_v": 0.03,
        "temp": "66°C",
        "soc": "88%",
        "odometer": "5,200 km"
    },
    "S1P-PUN-7731 (MH-12-ZT-7731)": {
        "model": "Ola S1 Pro Gen 1",
        "dtc": "ERR_CHG_LOCK_208",
        "delta_v": 0.08,
        "temp": "29°C",
        "soc": "4%",
        "odometer": "31,800 km"
    },
    "S1A-HYD-5514 (TS-09-EV-5514)": {
        "model": "Ola S1 Air",
        "dtc": "ERR_VCPU_OS_771",
        "delta_v": 0.02,
        "temp": "33°C",
        "soc": "51%",
        "odometer": "12,650 km"
    }
}

# 3. REALISTIC OLA SERVICE HUBS (PAN-INDIA SPREAD)
SERVICE_HUBS = [
    "Bengaluru — Koramangala Mega Service Hub",
    "Bengaluru — Whitefield Experience & Care Center",
    "Mumbai — Andheri East Hub (MIDC)",
    "Mumbai — Powai Saki Vihar Service Station",
    "Delhi NCR — Okhla Phase III Service Center",
    "Delhi NCR — Gurugram Sector 29 Hub",
    "Pune — Wakad Regional Service Hub",
    "Hyderabad — Gachibowli High-Tech Service Bay",
    "Chennai — Guindy Industrial Care Hub"
]

# 4. ERP CENTRAL INVENTORY
MOCK_INVENTORY = {
    "WH-S1-04": {"name": "Battery Wiring Harness Assembly", "stock": 4, "bay": "Rack B-03", "status": "In Stock"},
    "MCU-BRG-01": {"name": "MCU Communication Bridge Board", "stock": 0, "bay": "N/A", "status": "Out of Stock (Transit from Hosur FutureFactory)"},
    "TIP-MOD-09": {"name": "Thermal Interface Pad Kit", "stock": 14, "bay": "Rack T-11", "status": "In Stock"},
    "CPL-ACT-12": {"name": "Charge Port Solenoid Actuator", "stock": 3, "bay": "Rack C-08", "status": "In Stock"},
    "VCPU-GEN2-03": {"name": "VCPU Gen 2 Central Telematics Unit", "stock": 1, "bay": "Rack E-01", "status": "Low Stock"}
}

def agentic_diagnose(vin_selection, hub_location):
    vehicle = MOCK_TELEMETRY[vin_selection]
    rag_text = retrieve_tsb(vehicle["dtc"])

    sku = next((k for k in MOCK_INVENTORY if k in rag_text), "UNKNOWN-SKU")
    inv = MOCK_INVENTORY.get(sku, {"name": "General Service Part", "stock": 0, "status": "Unavailable", "bay": "N/A"})

    trace = [
        f"🔍 [CAN Ingestion] Vehicle: {vehicle['model']} | Odo: {vehicle['odometer']} | SoC: {vehicle['soc']}",
        f"📡 [Telemetry Scan] DTC: {vehicle['dtc']} | Voltage Delta: {vehicle['delta_v']}V | Temp: {vehicle['temp']}",
        f"📚 [Vector RAG Retrieval] Queried MoveOS engineering manuals -> Matched protocol for '{vehicle['dtc']}'",
        f"⚙️ [Hub ERP Integration] Queried inventory for part SKU '{sku}' at {hub_location}",
        f"📦 [ERP Observation] Item: {inv['name']} | Status: {inv['status']} ({inv['stock']} units available at {inv['bay']})"
    ]

    ticket = f"""### 🛠️ OLA SERVICE WORK ORDER (MOVEOS DISPATCH)
**Model:** {vehicle['model']} | **Telemetry Identifier:** `{vin_selection.split(' ')[0]}`
**Service Hub Location:** {hub_location}
**Primary Fault Signature:** `{vehicle['dtc']}` (Pack Temp: {vehicle['temp']} | Voltage Delta: {vehicle['delta_v']}V)

---

#### 1. Prescribed SOP Action Protocol:
> {rag_text}

#### 2. Warehouse Bill of Materials (BOM):
* **Part Required:** {inv['name']} (`{sku}`)
* **Warehouse Bay:** {inv['bay']}
* **Availability:** {'✅ ' + inv['status'] + ' (' + str(inv['stock']) + ' units)' if inv['stock'] > 0 else '⚠️ ' + inv['status']}

#### 3. Service Bay Sign-Off Checklist:
* [ ] Verify auxiliary 12V voltage stabilizes $\ge$ 12.6V
* [ ] Flash/verify MoveOS diagnostic clearance for `{vehicle['dtc']}`
* [ ] Complete road test and log telemetry back to Ola Cloud
"""
    return "\n\n".join(trace), rag_text, ticket

# 5. GRADIO USER INTERFACE
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# ⚡ Ola Electric — Service Bay Diagnostic Copilot")
    gr.Markdown("Agentic RAG pipeline connecting real-time MoveOS CAN telemetry, technical service bulletins (TSBs), and regional hub inventory.")

    with gr.Row():
        with gr.Column(scale=1):
            vin = gr.Dropdown(
                choices=list(MOCK_TELEMETRY.keys()),
                value=list(MOCK_TELEMETRY.keys())[0],
                label="Vehicle In Bay (CAN Identifier)"
            )
            hub = gr.Dropdown(
                choices=SERVICE_HUBS,
                value=SERVICE_HUBS[0],
                label="Assigned Ola Service Hub"
            )
            btn = gr.Button("⚡ Run Autonomous Triage", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📋 Technician Work Order"):
                    out_ticket = gr.Markdown()
                with gr.TabItem("🧠 Agent Reasoning Loop"):
                    out_trace = gr.Textbox(lines=7, label="Autonomous Tool-Call Trace")
                with gr.TabItem("📖 Retrieved Engineering SOP"):
                    out_rag = gr.Textbox(lines=4, label="Grounded MoveOS TSB Manual")

    btn.click(agentic_diagnose, inputs=[vin, hub], outputs=[out_trace, out_rag, out_ticket])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
