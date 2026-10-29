from pathlib import Path

for f in ["v3/scripts/generate_temporal_figures.py", "scripts/generate_temporal_figures.py"]:
    text = Path(f).read_text(encoding="utf-8")
    old = 'plot_figure2_violin(args.input, args.output / "fig2_permutation_collapse.pdf")\n    plot_figure3_factorial_and_ablations(args.input, args.output / "fig3_factorial_ablations.pdf")'
    new = '''# Figure 2: Factorial & Ablations (save under both filenames for compatibility)
    plot_figure3_factorial_and_ablations(args.input, args.output / "fig2_factorial_ablations.pdf")
    plot_figure3_factorial_and_ablations(args.input, args.output / "fig3_factorial_ablations.pdf")
    
    # Figure 3: Surrogate Permutations Violin (save under both filenames for compatibility)
    plot_figure2_violin(args.input, args.output / "fig3_permutation_collapse.pdf")
    plot_figure2_violin(args.input, args.output / "fig2_permutation_collapse.pdf")'''
    if old in text:
        text = text.replace(old, new)
        Path(f).write_text(text, encoding="utf-8")
        print("Updated", f)
    else:
        print("Old target not found, checking with CRLF...")
        old_crlf = old.replace("\n", "\r\n")
        new_crlf = new.replace("\n", "\r\n")
        if old_crlf in text:
            text = text.replace(old_crlf, new_crlf)
            Path(f).write_text(text, encoding="utf-8")
            print("Updated with CRLF", f)
        else:
            print("Failed to find target in", f)
