"""Fix string escapes in update_figure_script.py."""
from pathlib import Path

script_source = Path("v3/scripts/test_redesign_figures.py").read_text(encoding="utf-8")

# Extract generate_fig2 and generate_fig3 bodies
import ast
tree = ast.parse(script_source)

# Let's inspect test_redesign_figures.py functions
with open("v3/scripts/test_redesign_figures.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# generate_fig2 is lines 13 to 142
# generate_fig3 is lines 145 to 216
fig2_code = "".join(lines[12:143]).replace("def generate_fig2(output_path: Path):", "def plot_figure3_factorial_and_ablations(results_dir: Path, output_path: Path):")
fig3_code = "".join(lines[144:217]).replace("def generate_fig3(output_path: Path):", "def plot_figure2_violin(results_dir: Path, output_path: Path):")

combined_code = fig3_code + "\n\n" + fig2_code

import re
for file_path in ["v3/scripts/generate_temporal_figures.py", "scripts/generate_temporal_figures.py"]:
    p = Path(file_path)
    content = p.read_text(encoding="utf-8")
    pattern = r"def plot_figure2_violin\(.*?\n(?=def export_latex_table1)"
    new_content, count = re.subn(pattern, lambda m: combined_code + "\n\n", content, flags=re.DOTALL)
    if count == 1:
        p.write_text(new_content, encoding="utf-8")
        print(f"Successfully updated {file_path}")
    else:
        print(f"Could not update {file_path}, pattern matched {count} times")
