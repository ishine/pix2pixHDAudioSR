import urllib.request
import os
import re 
from tqdm import tqdm
import wget
os.chdir('./')#改变当前路径
#refiles=open('speech_files_path.txt','w+')#存储所有下载连接
mainpath='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/'
def gettgz(url):  
    page=urllib.request.urlopen(url)
    html=page.read().decode('utf-8')
    reg=r'href=".*\.tgz"'  
    tgzre=re.compile(reg)  
    tgzlist=re.findall(tgzre,html)  #找到所有.tgz文件
    i = 0
    for f in tqdm(tgzlist):
        if i<=1050:
            i = i+1
            continue
        filename=f.replace('href="','')
        filename=filename.replace('"','')
        print('正在下载：'+filename) #提示正在下载的文件
        downfile=f.replace('href="',mainpath)
        downfile=downfile.replace('"','') #得到每个文件的完整连接
        #req = urllib.Request(downfile)  #下载文件 
        wget.download(downfile, out=filename)
        i = i + 1

html=gettgz(mainpath)  
#refiles.close()