from pathlib import Path
packages=['PRM'] + [str(x) for x in Path('PRM').rglob('*/') if x.is_dir() and '__' not in str(x)],
print(packages)