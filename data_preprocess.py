from PIL import Image, ImageOps
import numpy as np
import os
from skimage import color, data, restoration
from scipy import ndimage
import av
import os
import progressbar
import glob

def sharpen(image,psf, display=True, quality = True):
    image_offset = image.copy()/255.
    image_offset += 1e-10
    sharpened = restoration.richardson_lucy(image_offset, psf, iterations=30)
    sharpened = sharpened
    
    if display:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        plt.gray()
        for a in (ax[0], ax[1]):
            a.axis('off')

        ax[0].imshow(image)
        ax[0].set_title('Original Data')

        ax[1].imshow(sharpened, vmin=image.min(), vmax=image.max())
        ax[1].set_title('Restoration using\nRichardson-Lucy')

        fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
        plt.show()

    if quality:
        original = Image.fromarray(np.uint8(image))
        sharp = Image.fromarray(np.uint8(sharpened*255.))
        original.save('original.png')
        sharp.save('sharp.png')
        return sharpened, get_score('original.png'), get_score('sharp.png')
    else:
        return sharpened, 0, 0
    
all_videos = sorted(glob.glob('video_dataset/*/*.MOV'))

bar = progressbar.ProgressBar(max_value=len(all_videos))
for idx,vid in enumerate(all_videos):
    bar.update(idx+1)
    v = av.open(vid)
    fold = vid.split('/')[1]
    save_dir = 'img_dataset/{}/{}'.format(fold,idx)
    set_save_dir = 'npy_dataset/{}/{}'.format(fold,idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(set_save_dir):
        os.makedirs(set_save_dir)
        
    video_set = np.zeros((100,224,224,3))
        
    sampling = np.linspace(0,v.streams.video[0].frames,100,dtype=int)
    for vidx,frame in enumerate(v.decode(video=0)):
        if vidx not in sampling:
            continue
        img_pil = frame.to_image()
        img_npy = np.array(img_pil)
        img_npy = np.transpose(img_npy,(1,0,2))
        img_pil = Image.fromarray(img_npy)
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        img_npy = np.array(img_pil)[50:-50,100:-25,:]
        img_pil = Image.fromarray(img_npy)
        img_pil = img_pil.resize((224,224))
        img_pil = np.array(img_pil)
        psf = np.random.beta(1,1,size=(3,3,1))
        sharpened, _, _ = sharpen(img_pil,psf,False,False)
        sharpened_img = Image.fromarray(np.uint8(sharpened*255.))
        video_set[np.where(sampling==vidx)[0][0]] = sharpened
        sharpened_img.save(os.path.join(save_dir,'{}.png'.format(vidx)))
    np.save(os.path.join(set_save_dir,'{}.npy'.format(idx)), video_set)