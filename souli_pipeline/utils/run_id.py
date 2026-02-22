import os
import datetime
import secrets

def get_run_id() -> str:
    rid = os.getenv("SOULI_RUN_ID")
    if rid:
        return rid
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{secrets.token_hex(3)}"
