from pathlib import Path
import subprocess

paths = [
    Path('v3/scripts/generate_temporal_figures.py'),
    Path('scripts/generate_temporal_figures.py')
]

for p in paths:
    text = p.read_text(encoding='utf-8').replace('\r\n', '\n')
    
    # In ax1, change ylim from (-0.5, 24.5) to (-1.0, 27.5)
    text = text.replace('ax1.set_ylim(-0.5, 24.5)', 'ax1.set_ylim(-1.0, 27.5)')
    
    # In ax3, change ylim from (-5.5, 0.50) to (-5.5, 0.70)
    text = text.replace('ax3.set_ylim(-5.5, 0.50)', 'ax3.set_ylim(-5.5, 0.70)')
    
    p.write_text(text.replace('\n', '\r\n'), encoding='utf-8')
    print(f"Refined {p}")
