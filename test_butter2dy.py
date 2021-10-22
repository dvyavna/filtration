#!/usr/bin/python3
from filters import butter2d_bp,imfft,imifft
from skimage import io, data
from matplotlib import pyplot as plt
import numpy as np
import argparse, os#, sys
from skimage.transform import resize
from mymisc import incr_oct,GammaDecode,GammaEncode
from myplot import Plot3dSurf
from numpy.fft import fft2, fftshift,ifft2,ifftshift
from skimage.color import rgb2ycbcr,ycbcr2rgb
#print(io.find_available_plugins())

def display_im(im):
    plt.figure()
    io.imshow(np.log(np.abs(imfft(im))))
    io.show()

def apply_butter2d_bp(img,low,high):
    filt=butter2d_bp(size=(img.shape[0],img.shape[1]),cutin=low,cutoff=high,n=10)
    #Сохранение постоянной составляющей
    filt[img.shape[0]//2,img.shape[1]//2]=1.

    #Plot3dSurf(filt)
    #display_im(filt)

    if len(img.shape)==3:
        conv_count=img.shape[2]
    else:
        conv_count=1
        img=np.expand_dims(img, axis=-1)
        #print(img.shape)
    res=np.zeros(img.shape)
    print('filt',np.min(filt),np.max(filt),np.sum(filt))
    for cnt in range(conv_count):
        avebr=np.mean(img[:,:,cnt])
        tmpplane=img[:,:,cnt]
        print('before',np.min(tmpplane),np.max(tmpplane),np.mean(tmpplane))
        res[:,:,cnt]=imifft(filt*imfft(tmpplane))
        #display_im(tmpplane)
        print('after',np.min(res[:,:,cnt]),np.max(res[:,:,cnt]),np.mean(res[:,:,cnt]))
    if conv_count==1:
        res=res[:,:,0]

    #print('all after',np.min(res),np.max(res),np.mean(res))
    return res#.astype(np.uint8)

#Инициализация##############################################################
dest_size=(256,256)
pict_size=256
freqs=pict_size//2
current_peak_freq=freqs
bands1st={} #настройки фильтров первого этапа в ц/изобр
spectr1st={} #настройки фильтров первого этапа в долях спектра, 0.5 максимум
ratio=8
while True :
    current_peak_freq=incr_oct(current_peak_freq,-1)
    #print(current_peak_freq)
    bands1st[current_peak_freq]=[incr_oct(current_peak_freq,-1),incr_oct(current_peak_freq,1)]
    spectr1st[current_peak_freq/freqs*0.5]=[bands1st[current_peak_freq][0]/freqs*0.5,
        bands1st[current_peak_freq][1]/freqs*0.5]
    if current_peak_freq==16: break
#print(bands1st)
#print(spectr1st)
#spectr1st[0.25]=[0.01, 0.2]
#exit(0)
#Инициализация##############################################################

parser = argparse.ArgumentParser(description='process one image')
parser.add_argument('--file', required=True, type=str, help='name of image file')
parser.add_argument('--dir', type=str, default='./', help='path to image dir')

imname=os.path.join(parser.parse_args().dir,parser.parse_args().file)
print(imname)

#Цветное изображение при открытии кодируется в uint8 (по крайней мере jpeg),
#а серое во float от 0 до 1
#image=io.imread(imname,as_gray=True)
#image=GammaDecode(io.imread(imname))
image=io.imread(imname)
print('image',image.shape)
image=image.astype(np.float32)
#image=resize(image,dest_size,anti_aliasing=True)
image=rgb2ycbcr(image)
#if np.max(image)<=1.: image*=255.
#print(image.dtype,np.max(image))
#exit(0)

for key in spectr1st.keys():
    #filtered=apply_butter2d_bp(image,spectr1st[key][0],spectr1st[key][1])
    filtered=np.zeros_like(image)
    filtered[:,:,0]=apply_butter2d_bp(image[:,:,0],spectr1st[key][0],spectr1st[key][1])
    filtered[:,:,1]=image[:,:,1]
    filtered[:,:,2]=image[:,:,2]
    filtered=ycbcr2rgb(filtered)
    new_imname=imname.split('.')
    new_imname='.'.join(new_imname[:len(new_imname)-1])+str(key)+'.'+new_imname[-1]
    print(new_imname)
    #io.imsave(new_imname,np.clip(GammaEncode(filtered),a_min=0.,a_max=255.).astype(np.uint8))
    print(np.max(filtered),np.min(filtered))
    io.imsave(new_imname,np.clip(filtered,a_min=0.,a_max=255.).astype(np.uint8))
