#!/usr/bin/env python3
"""Test SPARC parser"""

from pathlib import Path

sparc_file = Path("SPARC_Lelli2016c.mrt.txt")
with open(sparc_file, 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Find data start
data_start = 0
for i, line in enumerate(lines):
    if 'CamB' in line and i > 90:
        data_start = i
        print(f"Found data start at line {i}")
        break

print(f"\nFirst few data lines:")
for i in range(data_start, min(data_start + 5, len(lines))):
    print(f"Line {i}: {lines[i].strip()}")

# Try parsing first galaxy
line = lines[data_start]
print(f"\nParsing line length: {len(line)}")
print(f"Line content: '{line}'")

# Show character positions
print("\nCharacter positions:")
print("0         1         2         3         4         5         6         7         8         9")
print("0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789")
print(line.rstrip())

# Try extracting fields
try:
    galaxy = {
        'name': line[0:11].strip(),
        'type': line[11:13].strip(),
        'distance': line[13:19].strip(),
        'L36': line[34:41].strip(),
        'scale_length': line[61:66].strip(),
    }
    print(f"\nParsed galaxy: {galaxy}")
except Exception as e:
    print(f"\nError parsing: {e}") 