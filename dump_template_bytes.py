import os
p = os.path.join(os.getcwd(), 'backend', 'frontend', 'templates', 'index.html')
print('path:', p)
with open(p, 'rb') as f:
    b = f.read()
print('len bytes:', len(b))
print(repr(b[:200]))
print('\n---TEXT PREVIEW---\n')
print(b.decode('utf-8', errors='replace')[:2000])
