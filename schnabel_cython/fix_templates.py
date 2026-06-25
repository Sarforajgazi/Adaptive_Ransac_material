import os
import glob
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    orig_content = content

    # Fix at(Dereference(...))
    content = re.sub(r'(?<!>)(?<!:)\bat\(Dereference\(', r'this->at(this->Dereference(', content)
    content = content.replace("split(at(", "split(this->at(")
    content = content.replace("split(this->this->at(", "split(this->at(")

    if content != orig_content:
        print(f"Fixed {os.path.basename(filepath)}")
        with open(filepath, 'w', encoding='latin-1') as f:
            f.write(content)

base_dir = "/Users/sarforajgazi/Ransac_material/Efficient-RANSAC-for-Point-Cloud-Shape-Detection"
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.h') or file.endswith('.hpp') or file.endswith('.cpp'):
            fix_file(os.path.join(root, file))

print("Done fixing templates.")
