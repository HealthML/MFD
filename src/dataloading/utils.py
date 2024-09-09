
import hashlib
import gzip


def md5(fname):
    hash_md5 = hashlib.md5()
    
    if fname.endswith('gz'):
        with gzip.open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    else:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    
    return hash_md5.hexdigest()