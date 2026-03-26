import json

nb = json.load(open('DDER/Training_DDER.ipynb', 'r', encoding='utf-8'))
# Print cells 13, 15, 20, 22, 23 in full
for idx in [13, 15, 20, 21, 22, 23, 24]:
    c = nb['cells'][idx]
    src = ''.join(c['source'])
    with open(f'nb_cell_{idx}.txt', 'w', encoding='utf-8') as f:
        f.write(src)
    print(f"Cell {idx}: {len(src)} chars written to nb_cell_{idx}.txt")
