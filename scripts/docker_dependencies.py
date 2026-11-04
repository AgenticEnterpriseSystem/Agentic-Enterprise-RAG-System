"""Select exact Linux Docker pins without a dependency resolver or model import."""

import argparse
from pathlib import Path
import re

_DENSE = {'torch', 'sentence-transformers', 'faiss-cpu', 'transformers', 'tokenizers',
          'huggingface-hub', 'hf-xet', 'safetensors', 'sympy', 'networkx', 'mpmath'}


def selected_pins(path, *, hybrid):
    pins = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        package, separator, marker = line.partition(';')
        if not separator or marker.strip() not in {'sys_platform == "darwin"', 'sys_platform == "linux"'}:
            raise ValueError('Docker lock requires explicit supported platform markers')
        if marker.strip() != 'sys_platform == "linux"':
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9.+_-]+)', package.strip())
        if match is None:
            raise ValueError('Docker packages require exact versions')
        name, version = match.groups()
        key = re.sub(r'[-_.]+', '-', name).lower()
        if key in pins:
            raise ValueError('Duplicate Docker dependency')
        if hybrid or key not in _DENSE:
            pins[key] = f'{name}=={version}'
    if not pins or (hybrid and pins.get('torch') != 'torch==2.14.1+cpu'):
        raise ValueError('Verified Docker CPU lock is unavailable')
    return tuple(pins[key] for key in sorted(pins))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path('requirements.lock.txt'))
    parser.add_argument('--hybrid', choices=('true', 'false'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text('\n'.join(pin for pin in selected_pins(args.lock, hybrid=args.hybrid == 'true')
                                     if not pin.startswith('torch==')) + '\n')


if __name__ == '__main__':
    main()
