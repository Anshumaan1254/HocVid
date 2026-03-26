import json
with open('DDER/Training_DDER.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('dder_train_dump.txt', 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        source = cell.get('source', [])
        if isinstance(source, list):
            out.write(''.join(source) + '\n')
        else:
            out.write(str(source) + '\n')
        out.write('\n# --- CELL BOUNDARY ---\n\n')
