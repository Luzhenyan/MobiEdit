import re

def extract_norms(filename):
    pattern = re.compile(r'^(.*\.norm):([ \d\.\-eE]+)')
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                results.append((m.group(1).strip(), float(m.group(2))))
    return results

def compare_norms(file1, file2, outfile):
    norms1 = extract_norms(file1)
    norms2 = extract_norms(file2)
    lines = []
    if len(norms1) != len(norms2):
        lines.append(f"Warning: norm line count mismatch ({len(norms1)} vs {len(norms2)}), comparing up to shortest.\n")
    minlen = min(len(norms1), len(norms2))
    for i in range(minlen):
        name1, val1 = norms1[i]
        name2, val2 = norms2[i]
        if name1 != name2:
            lines.append(f"[Line {i+1}] Warning: Different norm names: {name1} vs {name2}\n")
        diff = val2 - val1
        lines.append(f"[{i+1}] {name1}: {val2:.8f} - {val1:.8f} = {diff:.8f}\n")
    with open(outfile, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'Diff output written to {outfile}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python compare_norm.py log1.txt log2.txt [output.txt]')
        exit(1)
    outfile = sys.argv[3] if len(sys.argv) >= 4 else "diff_output.txt"
    compare_norms(sys.argv[1], sys.argv[2], outfile)