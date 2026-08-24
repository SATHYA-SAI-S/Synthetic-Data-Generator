import os

with open('requirements.txt', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('opacus\n', 'opacus==1.4.1\n')

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(content)
