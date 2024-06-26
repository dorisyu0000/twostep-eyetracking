from ast import literal_eval
from glob import glob
import re
os.makedirs('data/exp/m2', exist_ok=True)
for fp in glob('data/exp/m2/*.json'):
    with open(fp) as f:
        data = f.read()
    with open(fp.replace('m2', 'm2-bad'), 'w') as f:

        f.write(data)

    fixed = re.sub(r"array\((\[[,\d. ]*\])\)", r'\g<1>', data)
    fixed = fixed.replace('(', '[').replace(')', ']').replace("'", '"')
    fixed = re.sub('<triggers.*>', 'null', fixed)
    fixed = fixed.replace('True', 'true').replace('False', 'false')
    json.loads(fixed)

    with open(fp, 'w') as f:
        f.write(fixed)

