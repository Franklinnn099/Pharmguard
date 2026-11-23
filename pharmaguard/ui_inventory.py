"""
Inventory tab for the PharmaGuard dashboard.

Staff register products against shelf zones, record stock movements and physical counts, and run a
reconciliation that cross-checks each zone's shortfall with the camera's REVIEW/ALERT events.
Formatting helpers are pure functions (unit-tested); ``build_inventory_tab`` wires them to Gradio.
"""

from html import escape
from typing import Any, List, Optional
import time

import gradio as gr

from pharmaguard.inventory import InventoryStore, ZoneReconciliation

STOCK_HEADERS = ["SKU", "Product", "Zone", "Price (GH¢)", "Expected qty", "Last count", "Counted at"]
RECON_HEADERS = ["Zone", "Status", "Units short", "Value short (GH¢)", "ALERTs", "REVIEWs", "Items short"]
STATUS_TEXT = {
    "CORROBORATED": "Shortfall + camera events: review the evidence clips",
    "UNEXPLAINED_LOSS": "Shortfall the camera did not flag: audit counts and procedures",
    "NO_SHORTFALL": "Camera events but stock complete: likely false alarms",
    "OK": "No shortfall, no events",
}
PERIODS = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}


def stock_rows(inv: InventoryStore) -> List[List[Any]]:
    return [[s.sku, s.name, s.zone, round(s.unit_price, 2), s.expected_qty,
             "" if s.last_counted_qty is None else s.last_counted_qty, s.last_count_at or "never"]
            for s in inv.stock_lines()]


def recon_rows(results: List[ZoneReconciliation]) -> List[List[Any]]:
    return [[r.zone, r.status, r.shortfall_units, r.shortfall_value, r.alerts, r.reviews,
             ", ".join(f"{i['name']} x{i['shortfall']}" for i in r.short_items)] for r in results]


def recon_summary_html(results: List[ZoneReconciliation]) -> str:
    if not results:
        return "<div style='font-size:13px'>No products or events in this period.</div>"
    lines = "".join(f"<li><b>{escape(r.zone)}</b>: {escape(r.status)} - {escape(STATUS_TEXT[r.status])}</li>"
                    for r in results)
    return (f"<div style='font-size:13px;line-height:1.6'><ul style='margin:0;padding-left:18px'>{lines}</ul>"
            "<div style='color:#94a3b8;font-size:12px'>Reasons for staff review only, never a finding of theft. "
            "Shortfalls come from counts taken in the period.</div></div>")


def run_reconciliation(inv: InventoryStore, event_store: Any, days: float,
                       exclude_false_alarms: bool, now: Optional[float] = None) -> List[ZoneReconciliation]:
    end = time.time() if now is None else now
    start = end - days * 86400
    events = event_store.zone_event_counts(start, end, exclude_false_alarms) if event_store is not None else {}
    return inv.reconcile(events, start, end)


def build_inventory_tab(pipeline: Any):
    """Create the Inventory tab contents (call inside a ``gr.Tab``). Returns (load_fn, inputs, outputs)."""
    data_dir = pipeline.config.recording.data_dir
    inv = InventoryStore(data_dir)
    pipeline.inventory = inv
    zone_names = [z.name for z in pipeline.config.zones] or ["Shelf"]

    gr.Markdown("Register products on the shelf zones the camera watches, record stock movements and counts, "
                "then reconcile stock against camera events.")
    stock_table = gr.Dataframe(headers=STOCK_HEADERS, value=[], interactive=False, max_height=260,
                               label="Stock (expected = last count + movements since)")
    with gr.Accordion("Add product", open=False):
        with gr.Row():
            sku_in = gr.Textbox(label="SKU", placeholder="PARA-500")
            name_in = gr.Textbox(label="Product name", placeholder="Paracetamol 500 mg x24")
            zone_in = gr.Dropdown(zone_names, value=zone_names[0], label="Shelf zone", allow_custom_value=True)
            price_in = gr.Number(label="Unit price (GH¢)", value=0, minimum=0)
            open_in = gr.Number(label="Opening qty", value=0, minimum=0, precision=0)
        add_btn = gr.Button("Add product", size="sm")
    with gr.Accordion("Stock movement / physical count", open=True):
        with gr.Row():
            sku_dd = gr.Dropdown([], label="Product (SKU)", allow_custom_value=True)
            kind_dd = gr.Dropdown([("Sale", "sale"), ("Restock", "restock"), ("Write-off (damaged/expired)", "writeoff"),
                                   ("Physical count", "count")], value="sale", label="Action")
            qty_in = gr.Number(label="Quantity (for a count: units on the shelf)", value=1, minimum=0, precision=0)
        mv_btn = gr.Button("Record", variant="primary", size="sm")
    feedback = gr.Markdown("")
    with gr.Row():
        period_dd = gr.Dropdown(list(PERIODS), value="Last 24 hours", label="Reconciliation period")
        excl_cb = gr.Checkbox(value=True, label="Ignore events staff marked as false alarm")
        recon_btn = gr.Button("Reconcile stock vs camera events", variant="primary", size="sm")
    recon_table = gr.Dataframe(headers=RECON_HEADERS, value=[], interactive=False, wrap=True, label="Reconciliation by zone")
    recon_html = gr.HTML("")

    def operator(request: Optional[gr.Request]) -> str:
        return (getattr(request, "username", None) or "operator") if request is not None else "operator"

    def skus():
        return gr.update(choices=[s.sku for s in inv.stock_lines()])

    def inventory_load():
        return stock_rows(inv), skus()

    def inventory_add(sku, name, zone, price, opening, request: gr.Request = None):
        try:
            inv.add_product(sku or "", name or "", zone or "", float(price or 0), int(opening or 0), operator(request))
        except ValueError as e:
            return f"Not added: {e}", gr.skip(), gr.skip()
        return f"Added **{escape(sku)}** to zone **{escape(zone)}**.", stock_rows(inv), skus()

    def inventory_move(sku, kind, qty, request: gr.Request = None):
        if not sku:
            return "Choose a product first.", gr.skip()
        try:
            if kind == "count":
                r = inv.record_count(sku, int(qty or 0), operator(request))
                msg = (f"Count saved: expected {r.expected_qty}, counted {r.counted_qty}. "
                       + (f"**{r.shortfall} unit(s) short.**" if r.shortfall > 0 else
                          f"{-r.shortfall} unit(s) over." if r.shortfall < 0 else "Matches."))
            else:
                inv.record_movement(sku, kind, int(qty or 0), operator(request))
                msg = f"Recorded {kind} of {int(qty)} for {escape(sku)}."
        except ValueError as e:
            return f"Not recorded: {e}", gr.skip()
        return msg, stock_rows(inv)

    def inventory_reconcile(period, excl):
        res = run_reconciliation(inv, pipeline.store, PERIODS.get(period, 1), bool(excl))
        return recon_rows(res), recon_summary_html(res)

    add_btn.click(inventory_add, [sku_in, name_in, zone_in, price_in, open_in], [feedback, stock_table, sku_dd])
    mv_btn.click(inventory_move, [sku_dd, kind_dd, qty_in], [feedback, stock_table])
    recon_btn.click(inventory_reconcile, [period_dd, excl_cb], [recon_table, recon_html])
    return inventory_load, [], [stock_table, sku_dd]
