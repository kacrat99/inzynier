import math

def graycode(x):
    return int(x)^(int(x)>>1)


def igraycode(x):
    
    if x == 0:
        return x
    m = int(math.ceil(math.log(x, 2)))+1
    i, j = x, 1
    while j < m:
        i = i ^ (x>>j)
        j += 1
    return i


def bits(n, width):
    
    assert n < 2**width
    bin = []
    for i in range(width):
        bin.insert(0, 1 if n&(1<<i) else 0)
    return bin


def bits2int(bits):
    
    n = 0
    for p, i in enumerate(reversed(bits)):
        n += i*2**p
    return n


def rrot(x, i, width):
    
    assert x < 2**width
    i = i%width
    x = (x>>i) | (x<<width-i)
    return x&(2**width-1)


def lrot(x, i, width):
    
    assert x < 2**width
    i = i%width
    x = (x<<i) | (x>>width-i)
    return x&(2**width-1)


def tsb(x, width):
    
    assert x < 2**width
    i = 0
    while x&1 and i <= width:
        x = x >> 1
        i += 1
    return i

#Ustawia bit i w liczbie całkowitej x o szerokości w na b.
def setbit(x, w, i, b):
   
    assert b in [1, 0]
    assert i < w
    if b:
        return x | 2**(w-i-1)
    else:
        return x & ~2**(w-i-1)

#Wyodrębnia zakres bitów jako liczbę całkowitą. 
def bitrange(x, width, start, end):
    
    return x >> (width-end) & ((2**(end-start))-1)

#Zwraca lokalną entropię bajtów dla lokalizacji w pliku.
def entropy(data, blocksize, offset, symbols=256):
    
    
    if offset < blocksize/2:
        start = 0
    elif offset > len(data)-blocksize/2:
        start = len(data)-blocksize/2
    else:
        start = offset-blocksize/2
    hist = {}
    for i in data[start:start+blocksize]:
        hist[i] = hist.get(i, 0) + 1
    base = min(blocksize, symbols)
    entropy = 0
    for i in hist.values():
        p = i/float(blocksize)
        
        entropy += (p * math.log(p, base))
    return -entropy
