"""Build the fictional development snapshot."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "pilot_snapshot"
ROOT.mkdir(parents=True, exist_ok=True)
BASE = "2026-11-01T00:00:00Z"


def fact(id, bubble, text, subject, relation, object, **kw):
    return {"unit_id": id, "item_id": "item_" + id, "bubble_id": bubble,
            "text": text, "start": 0, "end": len(text), "published_at": BASE,
            "available_at": BASE, "subject": subject, "relation": relation, "object": object,
            "global_entities": [subject, object], **kw}


items = [
    fact("f01", "finance", "Oriole Labs has its headquarters in Meridian.", "oriole", "headquarters", "meridian"),
    fact("f02", "research", "Tern Institute is led by Dr Sera Wynn.", "tern", "director", "sera"),
    fact("f03", "news", "Mira Vale announced the Lumen relay project.", "mira", "announced", "lumen"),
    fact("f04", "news", "The Lumen relay project has Oriole Labs as its sponsor.", "lumen", "sponsor", "oriole"),
    fact("f05", "research", "The Aster algorithm was developed at Tern Institute.", "aster", "developed_at", "tern"),
    fact("f06", "finance", "Oriole Labs funds Tern Institute through the Cedar grant.", "oriole", "funds", "tern"),
    fact("f07", "finance", "Until 5 November 2026 Oriole Labs had Niko Hale as CEO.", "oriole", "ceo", "niko",
         valid_from=BASE, valid_to="2026-11-05T00:00:00Z"),
    fact("f08", "finance", "From 5 November 2026 Oriole Labs has Leda Moss as CEO.", "oriole", "ceo", "leda",
         published_at="2026-11-05T00:00:00Z", available_at="2026-11-05T00:01:00Z", valid_from="2026-11-05T00:00:00Z"),
    fact("f09", "news", "The Lumen launch venue is Harbor Hall.", "lumen", "venue", "harbor",
         published_at="2026-11-08T09:00:00Z", available_at="2026-11-08T09:02:00Z", valid_from="2026-11-08T09:00:00Z"),
    fact("f10", "research", "Tern Institute released the Quill paper on 9 November 2026.", "tern", "released", "quill",
         published_at="2026-11-09T10:00:00Z", available_at="2026-11-09T10:03:00Z", valid_from="2026-11-09T10:00:00Z"),
]


def question(id, kind, text, bubbles, supports, answer, at="2026-11-10T12:00:00Z", known=None):
    return {"id": id, "kind": kind, "question": text, "required_bubbles": bubbles,
            "required_supports": supports, "reference_answer": answer, "answerable": bool(supports),
            "scope": {"start": at}, "known_at": known or at, "split": "development_only"}


questions = [
    question("P01", "single_bubble", "Where are the headquarters of Oriole Labs?", ["finance"], ["f01"], "Meridian"),
    question("P02", "single_bubble", "Who leads Tern Institute?", ["research"], ["f02"], "Dr Sera Wynn"),
    question("P03", "single_bubble", "What project did Mira Vale announce?", ["news"], ["f03"], "Lumen relay"),
    question("P04", "multi_hop", "In which city is the sponsor of the Lumen relay project headquartered?", ["news", "finance"], ["f04", "f01"], "Meridian"),
    question("P05", "multi_hop", "Who leads the institute funded by Oriole Labs?", ["finance", "research"], ["f06", "f02"], "Dr Sera Wynn"),
    question("P06", "multi_hop", "Which company funds the institute that developed the Aster algorithm?", ["research", "finance"], ["f05", "f06"], "Oriole Labs"),
    question("P07", "temporal", "Who was CEO of Oriole Labs on 4 November 2026?", ["finance"], ["f07"], "Niko Hale", "2026-11-04T12:00:00Z"),
    question("P08", "temporal", "Who was CEO of Oriole Labs on 6 November 2026?", ["finance"], ["f08"], "Leda Moss", "2026-11-06T12:00:00Z"),
    question("P09", "temporal", "What CEO appointment was valid at Oriole Labs at the start of 5 November?", ["finance"], ["f08"], "Leda Moss", "2026-11-05T00:00:00Z", "2026-11-06T12:00:00Z"),
    question("P10", "freshness", "Where is the Lumen launch venue?", ["news"], [], "Not yet available", "2026-11-08T09:01:00Z"),
    question("P11", "freshness", "Where is the Lumen launch venue?", ["news"], ["f09"], "Harbor Hall", "2026-11-08T09:03:00Z"),
    question("P12", "freshness", "Which paper has Tern Institute released?", ["research"], ["f10"], "Quill", "2026-11-09T10:04:00Z"),
]
qrels = {q["id"]: {unit: [unit] for unit in q["required_supports"]} for q in questions}
for q in questions:
    for unit in q["required_supports"]:
        qrels[q["id"]]["item_" + unit + ":part1"] = [unit]

for name, rows in (("items.jsonl", items), ("questions.jsonl", questions)):
    (ROOT / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
(ROOT / "qrels.json").write_text(json.dumps(qrels, indent=2), encoding="utf-8", newline="\n")
manifest = {"name": "fictional_pilot_v1", "version": 1, "split": "development_only",
            "fictional": True, "frozen_at": "2026-11-10T12:00:00Z",
            "item_count": len(items), "question_count": len(questions),
            "sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                       for n in ("items.jsonl", "questions.jsonl", "qrels.json")}}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
print(json.dumps({"snapshot": str(ROOT), "items": len(items), "questions": len(questions)}))
