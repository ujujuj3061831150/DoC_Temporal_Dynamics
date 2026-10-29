import fitz
import sys
from pathlib import Path

def verify_layout(pdf_path, expected_pages=None):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    if expected_pages is None:
        if "supplementary" in str(pdf_path).lower():
            expected_pages = 2
        else:
            expected_pages = 9
    assert total_pages == expected_pages, f"Expected {expected_pages} pages, got {total_pages}"

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[4].strip()]
        if not text_blocks:
            continue
        y_max = max(b[3] for b in text_blocks)
        y_min = min(b[1] for b in text_blocks)
        print(f"Page {i+1}: y_min={y_min:.1f}, y_max={y_max:.1f}")

    # Check last page column balance (page 9 for main, page 2 for supplementary)
    last_page = doc[total_pages - 1]
    blocks = last_page.get_text("blocks")
    mid_x = last_page.rect.width / 2
    col1_blocks = [b for b in blocks if b[0] < mid_x and b[4].strip()]
    col2_blocks = [b for b in blocks if b[0] >= mid_x and b[4].strip()]
    col1_bottom = max((b[3] for b in col1_blocks), default=0)
    col2_bottom = max((b[3] for b in col2_blocks), default=0)
    col_diff = abs(col1_bottom - col2_bottom)
    print(f"Page {total_pages}: col1_bottom={col1_bottom:.1f}, col2_bottom={col2_bottom:.1f}, diff={col_diff:.1f} pt")
    if expected_pages == 9:
        assert col_diff <= 35.0, f"Page {total_pages} column diff {col_diff:.1f} pt exceeds 35 pt threshold!"

    # Check word/line overlaps
    total_overlaps = 0
    for i, page in enumerate(doc):
        words = page.get_text("words")
        lines = {}
        for w in words:
            key = (w[5], w[6])
            if key not in lines:
                lines[key] = []
            lines[key].append(w)

        line_rects = []
        for key, l_words in lines.items():
            x0 = min(w[0] for w in l_words)
            y0 = min(w[1] for w in l_words)
            x1 = max(w[2] for w in l_words)
            y1 = max(w[3] for w in l_words)
            line_rects.append((fitz.Rect(x0, y0, x1, y1), " ".join(w[4] for w in l_words)))

        page_overlaps = []
        for idx1 in range(len(line_rects)):
            r1, t1 = line_rects[idx1]
            for idx2 in range(idx1 + 1, len(line_rects)):
                r2, t2 = line_rects[idx2]
                if min(r1.x1, r2.x1) - max(r1.x0, r2.x0) > 30:
                    v_overlap = min(r1.y1, r2.y1) - max(r1.y0, r2.y0)
                    if v_overlap > 2.0:
                        page_overlaps.append((v_overlap, t1, t2))
        if page_overlaps:
            print(f"WARNING: Page {i+1} has {len(page_overlaps)} overlaps:")
            for vo, t1, t2 in page_overlaps:
                print(f"  Overlap {vo:.2f}pt:")
                print(f"    Line A: {t1.encode('ascii', 'replace').decode()}")
                print(f"    Line B: {t2.encode('ascii', 'replace').decode()}")
            total_overlaps += len(page_overlaps)

    print(f"Total line overlaps across all pages: {total_overlaps}")
    assert total_overlaps == 0, f"Detected {total_overlaps} text overlaps!"
    print("ALL LAYOUT CRITERIA VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "v3/manuscript_temporal/manuscript.pdf"
    exp = int(sys.argv[2]) if len(sys.argv) > 2 else None
    verify_layout(pdf, exp)
