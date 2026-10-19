"""Check the public release manifest without reading external experiment files."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    checked = 0
    for name, item in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('Manifest path outside repository')
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Release file changed: ' + name)
        if name.endswith('.py'):
            ast.parse(data, filename=name)
        checked += 1
    print(json.dumps({'status': 'ok', 'files_verified': checked, 'gpu_training_rerun': False}))


if __name__ == '__main__':
    main()
