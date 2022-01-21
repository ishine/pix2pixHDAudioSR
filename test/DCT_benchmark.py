import torch
from dct.dct_lee import DCT
import time
dct = DCT().cuda()
dur = []
with torch.cuda.device(1):
    for i in range(10):
        a = torch.rand((16,1,512,512),device='cuda')
        tik = time.time()
        b = dct(a)
        tok  = time.time()
        dur.append(tok-tik)
print(dur)
