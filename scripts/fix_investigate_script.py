# -*- coding: utf-8 -*-
from pathlib import Path

for path in ['v3/scripts/investigate_reviewer_points.py', 'scripts/investigate_reviewer_points.py']:
    p = Path(path)
    if not p.exists():
        continue
    with open(p, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')
    
    old_block = '''REPO_ROOT = Path(".")
import sys
sys.path.append(str(REPO_ROOT / "v3/src"))'''
    new_block = '''import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "v3" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))'''
    
    if old_block in text:
        text = text.replace(old_block, new_block)
        with open(p, 'w', encoding='utf-8', newline='\r\n') as f:
            f.write(text)
        print(f"Fixed portable import in {path}")
    else:
        print(f"old_block not found in {path}")
