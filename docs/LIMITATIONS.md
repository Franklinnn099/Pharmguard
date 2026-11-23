# PharmaGuard: Known Limitations (for the report and the defence)

PharmaGuard flags interaction sequences **for staff review**. It does not identify theft or intent. Each limitation
below is a consequence of a design choice, stated with the reason and the mitigation.

## Vision

| Limitation | Why | Mitigation / future work |
|---|---|---|
| **Only COCO bag types are detected** (backpack, handbag, suitcase). Polythene bags, sacks, sleeves, umbrellas, coats are not | No custom training in scope (TECHNICAL_DECISIONS §7); pretrained COCO classes only | Pocket path still applies. Future: fine-tune a detector on local bag types (polythene "rubber" bags are common in Ghana) |
| **Pocket check is off when hips are not visible** | The rule needs hip keypoints; guessing would cause false alarms | Camera placement (DEMO_STAGING §1). Reported as a miss mode |
| **Arm hidden from the camera** (turned away, behind the body) is missed | The system tracks wrists, not products | One camera angle per shelf; future: second camera |
| **Small items palmed without arm movement** are missed | Same reason | Inventory reconciliation catches the shortfall later (UNEXPLAINED_LOSS) |
| **Item held at the hip > 4 s raises an ALERT** even if paid for later | Trade-off for a realistic trigger (§13) | Staff review; measured in evaluation scenario F |
| **Single camera, single operator** | One shared pipeline (PROJECT_STATUS R8) | Single-operator demo |
| **Evaluated on staged clips only** | No pharmacy access; filming real customers needs consent and ethics approval | Mock-shelf evaluation (docs/EVALUATION_PLAN.md); a real pharmacy pilot is future work |
| **Layout differs from Ghanaian counter-service pharmacies**, where most stock sits behind the counter | The mock shelf models the open OTC/cosmetics section, where customers can reach stock | Scope statement: open-shelf sections only |

## Inventory

| Limitation | Why | Mitigation |
|---|---|---|
| Sales, restocks and counts are entered by hand in the dashboard | No POS integration in scope | Future: import sales from the pharmacy's POS |
| Shortfalls are only known when someone counts | Expected stock is a book figure until counted | Daily count of high-risk zones |
| A shortfall is linked to camera events by **zone and period**, not by item | The camera sees hands, not products | Status names say "review", never "theft" |
| Staff removing stock outside procedure is not detected by the camera | Staff handle stock all day; the behaviour rules are for customers | Shows up as UNEXPLAINED_LOSS in reconciliation |

## Performance

| Limitation | Why | Mitigation |
|---|---|---|
| All measured speeds are CPU on the dev laptop | Colab T4 run not yet done | PROJECT_STATUS §3 checklist before the demo |
| Over the `gradio.live` tunnel FPS depends on network round trip | Each frame is an HTTP round trip | Fast stream with frames in flight (§12); recorded fallback video |
