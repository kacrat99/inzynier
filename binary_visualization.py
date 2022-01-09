from PIL import Image, ImageDraw
import os.path, math, string, sys
import hilber_curve

class _Color:
    def __init__(self, data):
        self.data, self.block = data, None
        self.data = data
        s = list(set(data))
        s.sort()
        self.symbol_map = {v : i for (i, v) in enumerate(s)}

    def __len__(self):
        return len(self.data)
    def point(self, x):
        if self.block and (self.block[0]<=x<self.block[1]):
            return self.block[2]
        else:
            return self.getPoint(x)


def drawmap_unrolled( size, csource, name):
    
    map = fs(2, size**2)
    c = Image.new("RGB", (size, size*4))
    cd = ImageDraw.Draw(c)
    step = len(csource)/float(len(map)*4)

    sofar = 0
    for quad in range(4):
        for i, p in enumerate(map):
            off = (i + (quad * size**2))
            color = csource.point(
                        int(off * step)
                    )
            x, y = tuple(p)
            cd.point(
                (x, y + (size * quad)),
                fill=tuple(color)
            )
            
    c.save(name)
    

def fs(dim,size):
    return hilber_curve.Hilbert.fromSize(dim,size)

class ColorEntropy(_Color):
    def getPoint(self, x):
        e = entropy(self.data, 32, x, len(self.symbol_map))
        # http://www.wolframalpha.com/input/?i=plot+%284%28x-0.5%29-4%28x-0.5%29**2%29**4+from+0.5+to+1
        def curve(v):
            f = (4*v - 4*v**2)**4
            f = max(f, 0)
            return f
        r = curve(e-0.5) if e > 0.5 else 0
        b = e**2
        return [
            int(255*r),
            0,
            int(255*b)
        ]
def entropy(data, blocksize, offset, symbols=256):
    
    if offset < blocksize/2:
        start = 0
    elif offset > len(data)-blocksize/2:
        start = len(data)-blocksize/2
    else:
        start = offset-blocksize/2
    hist = {}
    for i in data[int(start):int(start+blocksize)]:
        hist[i] = hist.get(i, 0) + 1
    base = min(blocksize, symbols)
    entropy = 0
    for i in hist.values():
        p = i/float(blocksize)
        
        entropy += (p * math.log(p, base))
    return -entropy

source = open("C:/Users/Kacper/Desktop/inzynier/apk/2.apk",encoding='utf-8',errors='ignore').read()
dst = "testing.png"

calc = ColorEntropy(source)

drawmap_unrolled( 256, calc, dst)