import tarfile

file_path = 'data/raw/aclImdb_v1.tar.gz'
extract_path = 'data/raw/aclImdb'

with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)

print("Dataset extracted!")
