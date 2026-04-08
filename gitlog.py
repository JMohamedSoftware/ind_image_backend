import subprocess
r = subprocess.run(
    ['git', 'log', '--pretty=%h %s', '-15'],
    capture_output=True, text=True,
    cwd=r'C:\Users\User\Desktop\image search and indexing\ind_image_backend'
)
for line in r.stdout.strip().split('\n'):
    print(line)
