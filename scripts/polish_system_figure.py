"""Read-only compatibility entry point. System styling is disabled by the user."""
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[1]
lock=root/'manuscript_temporal/figures/system_workflow.lock.json'
for name,expected in json.loads(lock.read_text(encoding='utf8'))['sha256'].items():
    assert hashlib.sha256((lock.parent/name).read_bytes()).hexdigest()==expected,name
print('Protected system framework verified; no changes made.')
