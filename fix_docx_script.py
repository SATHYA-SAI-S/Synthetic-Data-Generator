import re
import pathlib

path = pathlib.Path('scripts/generate_phase_docx.py')
content = path.read_text(encoding='utf-8')

# 1. Fix line 11: broken docx path string closing quote
content = content.replace(
    'OUTPUT_PATH = os.path.join("docs", "Phase_Explanatory_Document.docx)\n',
    'OUTPUT_PATH = os.path.join("docs", "Phase_Explanatory_Document.docx")\n'
)

# 2. Fix all broken multi-line add_para( ... " ) patterns
# The regex before broke lines ending with `")` by removing the quote.
# We need to find lines that now incorrectly end with `)` where a quote was removed.
# Pattern: line starts with optional whitespace, then a closing paren, line ends.
# Original final lines of add_para calls look like `)` alone on a line.

# Yes, the fix is simply: the regex already converted `")` to `)` for those ending lines,
# but it ALSO converted `")` in other contexts. So we need to find strings that
# became unterminated: e.g. `".docx)` should be `".docx")`.
# All non-symbol lines that start a string but end with `)` without a closing quote.

# Simpler: find any line that starts with `"` (or whitespace + `"`) and ends with `)`
# but does NOT end with `")`. Those are lines where the closing quote was eaten.
result_lines = []
for line in content.split('\n'):
    stripped = line.strip()
    # If the line contains a string opener but ends with ) without a closing quote
    # before the ), and the ) is at the very end.
    # Heuristic: line ends with `)` but the content before contains an unmatched `"`.
    if stripped.endswith(')') and stripped.count('"') % 2 == 1:
        # Ends with ) but has odd number of quotes - repair: add `)` at the very end
        # Actually the original form was `")` -> we need to restore the '"' before ')'
        # Insert the quote right before the final ')'
        line = line.rstrip()
        if line.endswith(')'):
            line = line[:-1] + '")' + '\n'
        else:
            line = line + '\n'
    result_lines.append(line)
content = ''.join(result_lines)

path.write_text(content, encoding='utf-8')
print('Fixed!')