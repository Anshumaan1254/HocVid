import json
with open('evoIR_aflb/models/res_fftb.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('res_fftb_dump.txt', 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        source = cell.get('source', [])
        if isinstance(source, list):
            out.write(''.join(source) + '\n')
        else:
            out.write(str(source) + '\n')
        out.write('\n# --- CELL BOUNDARY ---\n\n')
