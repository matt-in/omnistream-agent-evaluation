"""Persist question outcomes and verify the identity of a resumed run."""

import json
import sqlite3
from contextlib import contextmanager


@contextmanager
def open_checkpoint(path, identity, *, resume):
    if resume and not path.is_file():
        raise FileNotFoundError("Resume requires a checkpoint from the same run")
    db = sqlite3.connect(path)
    try:
        if resume:
            record = db.execute("SELECT settings FROM run_identity").fetchone()
            if record is None or json.loads(record[0]) != identity:
                raise ValueError("Checkpoint does not match the snapshot, configuration, adapter revision or code")
        else:
            with db:
                db.execute("CREATE TABLE run_identity (settings TEXT NOT NULL)")
                db.execute("INSERT INTO run_identity VALUES (?)", (json.dumps(identity),))
                db.execute("""CREATE TABLE outcomes (
                    question_id TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    measured INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    trace TEXT NOT NULL,
                    PRIMARY KEY (question_id, configuration)
                )""")
        yield db
    finally:
        db.close()


def saved_outcomes(db):
    rows = db.execute("SELECT result, trace FROM outcomes ORDER BY configuration, question_id")
    return [(json.loads(result), json.loads(trace)) for result, trace in rows]


def save_outcome(db, row, trace):
    with db:
        db.execute("""INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (question_id, configuration) DO UPDATE SET
              measured = excluded.measured, result = excluded.result, trace = excluded.trace
            WHERE outcomes.measured = 0""",
            (row["question_id"], row["configuration"], row["measured"],
             json.dumps(row, ensure_ascii=False), json.dumps(trace, ensure_ascii=False)),
        )
