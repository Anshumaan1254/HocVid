import json
import os
import glob
import re

def fix_notebook(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                original_line = line
                
                # BUG 1: abs -> .real
                if 'low = low.real' in line:
                    line = line.replace('low = low.real', 'low = low.real')
                
                # BUG 1b: norm='ortho' -> norm='ortho'
                if "norm='ortho'" in line:
                    line = line.replace("norm='ortho'", "norm='ortho'")
                if 'norm="ortho"' in line:
                    line = line.replace('norm="ortho"', 'norm="ortho"')
                
                # BUG 3: h//n -> max(1, h//n)
                # Looking for: max(1, h // self.n) or w // self.n or similar
                if 'h // self.n' in line and 'max' not in line:
                    line = line.replace('h // self.n', 'max(1, h // self.n)')
                if 'w // self.n' in line and 'max' not in line:
                    line = line.replace('w // self.n', 'max(1, w // self.n)')
                if ':h//' in line and 'max' not in line:
                    line = re.sub(r':h//([a-zA-Z0-9_\.]+)', r':max(1, h//\1)', line)
                if ':w//' in line and 'max' not in line:
                    line = re.sub(r':w//([a-zA-Z0-9_\.]+)', r':max(1, w//\1)', line)
                
                # BUG 5: num_heads=4 -> 4
                if 'num_heads=4' in line:
                    line = line.replace('num_heads=4', 'num_heads=4')
                
                # BUG 6: Sigmoid multiply
                if 'x = torch.sigmoid(self.pc(p1)) * p1' in line:
                    line = line.replace('+ p1', '* p1')
                if 'prompt = torch.sigmoid(prompt2)' in line:
                    pass # Keep
                if 'x = prompt * prompt1' in line:
                    line = line.replace('x = prompt * prompt1', 'x = prompt * prompt1')
                if 'x=torch.sigmoid(s.pc(p1))*p1' in line:
                    line = line.replace('+p1', '*p1')
                if 'x=torch.sigmoid(p2)*p1' in line:
                    line = line.replace('+p1', '*p1')
                    
                # BUG 2: num_fft_blocks[1] -> num_fft_blocks[3]
                if 'latent_fft' in line and 'num_fft_blocks[3]' in line:
                    line = line.replace('num_fft_blocks[1]', 'num_fft_blocks[3]')
                    
                # DEAD CODE: Remove self.conv in AFLB
                # Only if it's the one that is unused (self.conv vs self.conv1)
                if 'self.conv =' in line and 'nn.Conv2d' in line and 'AFLB' not in line:
                    # Actually, the user says "Remove self.conv from AFLB.__init__"
                    pass # We will handle this with regex or multi-line if needed, but actually we can just regex it if it's exactly `self.conv = ...` in AFLB
                
                # train.ipynb: Clamp output before MS-SSIM
                if 'ms_ssim' in line and 'restored,' in line and 'clamp' not in line:
                    line = line.replace('restored,', 'restored.clamp(0, 1),')
                if 'ms_ssim(' in line and 'restored ' in line and 'clamp' not in line: # Fallback
                     line = line.replace('restored', 'restored.clamp(0,1)')
                     
                # test.ipynb: count.clamp
                if 'restored = restored / count' in line and 'clamp' not in line:
                    line = line.replace('count', 'count.clamp(min=1e-5)')
                if 'restored /= count' in line and 'clamp' not in line:
                    line = line.replace('restored /= count', 'restored = restored / count.clamp(min=1e-5)')
                
                # MS-SSIM betas and kernel
                if 'kernel_size=5' in line and 'MS_SSIM' in line:
                     line = line.replace('kernel_size=5', 'kernel_size=11')
                     
                # Add to new source
                new_source.append(line)
            
            # Additional multi-line processing per cell
            cell_text = "".join(new_source)
            if 'self.conv = ' in cell_text and 'self.conv1 =' in cell_text:
                # Remove self.conv line completely
                new_source = [l for l in new_source if not l.strip().startswith('self.conv =')]
                
            # test.ipynb kwarg fixes
            if 'test.ipynb' in filepath:
                modified_source = []
                for l in new_source:
                    if 'AdaIR(' in l or 'AdaIR (' in l:
                        l = l.replace('nb=', 'num_blocks=')
                        l = l.replace('nf=', 'num_fft_blocks=')
                        l = l.replace('nrb=', 'num_refinement_blocks=')
                    modified_source.append(l)
                new_source = modified_source
                
            # eos.ipynb / train.ipynb cache clipping
            if ('train.ipynb' in filepath or 'eos.ipynb' in filepath) and 'self.cache' in cell_text:
                # This requires moving `if len(self.cache) > self.pop_size * 2: self.cache = self.cache[-self.pop_size*2:]`
                # We'll just append it to update_loss_weights or cache appending method.
                modified_source = []
                for l in new_source:
                    modified_source.append(l)
                    if 'self.cache.append(' in l:
                        indent = l[:len(l) - len(l.lstrip())]
                        modified_source.append(indent + "if len(self.cache) > self.pop_size * 2:\n")
                        modified_source.append(indent + "    self.cache = self.cache[-self.pop_size * 2:]\n")
                new_source = modified_source

            if cell['source'] != new_source:
                cell['source'] = new_source
                modified = True
                
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed: {filepath}")

def fix_python_files(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified_source = []
    modified = False
    for line in lines:
        old_line = line
        
        # BUG 1
        if 'low = low.real' in line: line = line.replace('low = low.real', 'low = low.real')
        
        # BUG 1b
        if "norm='ortho'" in line: line = line.replace("norm='ortho'", "norm='ortho'")
        if 'norm="ortho"' in line: line = line.replace('norm="ortho"', 'norm="ortho"')
        
        # BUG 3
        if 'h // self.n' in line and 'max' not in line: line = line.replace('h // self.n', 'max(1, h // self.n)')
        if 'w // self.n' in line and 'max' not in line: line = line.replace('w // self.n', 'max(1, w // self.n)')
        if ':h//' in line and 'max' not in line: line = re.sub(r':h//([a-zA-Z0-9_\.]+)', r':max(1, h//\1)', line)
        if ':w//' in line and 'max' not in line: line = re.sub(r':w//([a-zA-Z0-9_\.]+)', r':max(1, w//\1)', line)
        
        # BUG 5
        if 'num_heads=4' in line: line = line.replace('num_heads=4', 'num_heads=4')
        
        # BUG 6
        if 'x = torch.sigmoid(self.pc(p1)) * p1' in line: line = line.replace('* p1', '* p1')
        if 'x = prompt * prompt1' in line: line = line.replace('x = prompt * prompt1', 'x = prompt * prompt1')
        if 'x=torch.sigmoid(s.pc(p1))*p1' in line: line = line.replace('*p1', '*p1')
        if 'x=torch.sigmoid(p2)*p1' in line: line = line.replace('*p1', '*p1')
        
        # BUG 2
        if 'latent_fft' in line and 'num_fft_blocks[3]' in line:
            line = line.replace('num_fft_blocks[1]', 'num_fft_blocks[3]')
            
        modified_source.append(line)
        if line != old_line:
            modified = True
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(modified_source)
        print(f"Fixed matching bugs in: {filepath}")

if __name__ == '__main__':
    notebooks = glob.glob('models/*.ipynb') + ['train.ipynb', 'test.ipynb']
    for nb in notebooks:
        fix_notebook(nb)
    
    python_files = glob.glob('*.py')
    for py in python_files:
        fix_python_files(py)
