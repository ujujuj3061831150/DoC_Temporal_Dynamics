from pathlib import Path

paths = [
    Path('v3/manuscript_temporal/references.bib'),
    Path('manuscript_temporal/references.bib')
]

for p in paths:
    text = p.read_text(encoding='utf-8')
    # replace BSTcontrol
    import re
    old_pattern = r'@IEEEtranBSTCTL\{IEEEexample:BSTcontrol,[^}]*\}'
    new_entry = """@IEEEtranBSTCTL{IEEEexample:BSTcontrol,
  CTLuse_forced_etal = "yes",
  CTLmax_names_forced_etal = "3",
  CTLnames_show_etal = "1",
  CTLdash_repeated_names = "no"
}"""
    assert re.search(old_pattern, text), f"Pattern not found in {p}"
    text = re.sub(old_pattern, new_entry, text)
    p.write_text(text, encoding='utf-8')
    print(f"Updated {p}")
