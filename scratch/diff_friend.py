import filecmp
import os

def compare_dirs(dir1, dir2):
    diffs = []
    def compare_recursive(d1, d2):
        cmp = filecmp.dircmp(d1, d2, ignore=['.git', '.venv310', '__pycache__', 'outputs', 'data', 'scratch', 'docs', '.pytest_cache'])
        for name in cmp.diff_files:
            diffs.append((os.path.relpath(os.path.join(d1, name), dir1), "Modified"))
        for name in cmp.right_only:
            diffs.append((os.path.relpath(os.path.join(d2, name), dir2), "New File"))
        for name in cmp.left_only:
            diffs.append((os.path.relpath(os.path.join(d1, name), dir1), "Deleted"))
        for common_dir in cmp.common_dirs:
            compare_recursive(os.path.join(d1, common_dir), os.path.join(d2, common_dir))
    compare_recursive(dir1, dir2)
    return diffs

d1 = r'e:\ADVT'
d2 = r'e:\ADVT\scratch\friend_code_py\ADVT'

diffs = compare_dirs(d1, d2)
if not diffs:
    print("No changes found in source files!")
else:
    for f, status in diffs:
        print(f"{status}: {f}")
