#!/usr/bin/env python3
"""Convert SVG to PNG and PDF with high quality settings."""

import cairosvg
import os
from pathlib import Path

# Input SVG file
svg_file = Path(__file__).parent / "main_diagram.svg"
output_dir = svg_file.parent

# Output files
png_file = output_dir / "main_diagram.png"
pdf_file = output_dir / "main_diagram.pdf"

print(f"Converting {svg_file.name}...")
print(f"  SVG dimensions: 5400x1860 (viewBox: 0 0 1800 620)")

# Convert to PNG with 300 DPI (high quality)
# DPI conversion: 300 DPI ≈ 11.81 px/mm
# For screen DPI of 96 px/inch: 300 DPI / 96 = 3.125 scale factor
print(f"\n📄 Creating PNG (300 DPI equivalent)...")
cairosvg.svg2png(
    url=str(svg_file),
    write_to=str(png_file),
    dpi=300,
)
png_size = png_file.stat().st_size / (1024**2)
print(f"✓ PNG created: {png_file.name} ({png_size:.2f} MB)")

# Convert to PDF with high quality
print(f"\n📋 Creating PDF...")
cairosvg.svg2pdf(
    url=str(svg_file),
    write_to=str(pdf_file),
)
pdf_size = pdf_file.stat().st_size / (1024**2)
print(f"✓ PDF created: {pdf_file.name} ({pdf_size:.2f} MB)")

print("\n✅ Conversion complete!")
print(f"\nGenerated files:")
print(f"  - {png_file}")
print(f"  - {pdf_file}")
