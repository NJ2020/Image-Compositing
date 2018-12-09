
# coding: utf-8

# In[ ]:


# Import relevant libraries
import numpy as np; np.random.seed(42); import tensorflow as tf; tf.set_random_seed(42);
import skimage.io as io; import cv2; import os; from pycocotools.coco import COCO; 
import matplotlib.pyplot as plt; from PIL import Image, ImageEnhance; 
import warnings; warnings.filterwarnings("ignore");

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#   Kelvin table is taken from link: http://www.andi-siess.de/portfolio/rgb-to-color-temperature

kelvin_table = {
    1600: (255, 115, 0),   1700: (255, 121, 0),   1800: (255, 126, 0),   1900: (255, 131, 0),   2000: (255, 138, 18), 
    2100: (255, 142, 33),  2200: (255, 147, 44),  2300: (255, 152, 54),  2400: (255, 157, 63),  2500: (255, 161, 72), 
    2600: (255, 165, 79),  2700: (255, 169, 87),  2800: (255, 173, 94),  2900: (255, 177, 101), 3000: (255, 180, 107), 
    3100: (255, 184, 114), 3200: (255, 187, 120), 3300: (255, 190, 126), 3400: (255, 193, 132), 3500: (255, 196, 137),
    3600: (255, 199, 143), 3700: (255, 201, 148), 3800: (255, 204, 153), 3900: (255, 206, 159), 4000: (255, 209, 163),
    4100: (255, 211, 168), 4200: (255, 213, 173), 4300: (255, 215, 177), 4400: (255, 217, 182), 4500: (255, 219, 186),
    4600: (255, 221, 190), 4700: (255, 223, 194), 4800: (255, 225, 198), 4900: (255, 227, 202)}

def change_temp_img(image, temp):

    """
    Arguments:
    image:     Original Image
    temp:      temperature of the image (should be in between 1600 and 4900)

    Returns:
    mod_image: Modified image
    """
 
    r, g, b = kelvin_table[temp];
    
#   Convert the image into array
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    matrix = ( r/255., 0., 0., 0., 0., g/255., 0., 0., 0., 0., b/255.0, 0.)
    mod_img = np.uint8(image.convert('RGB', matrix))
    
    return mod_img

def change_contrast_img(image, level):
  
    """
    Arguments:
    image:     Original Image
    level:     Level by which to change the contrast of the image

    Returns:
    mod_image: Modified image
    """
    
#   Convert the image into array
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    factor = (259*(level + 255))/(255*(259 - level))
    
    def contrast(c):
        return 128 + factor * (c - 128)
    
    mod_img = np.uint8(np.array(image.point(contrast)))
    
    return mod_img


# In[ ]:


def image_stats(image):
    """
    Arguments: 
    image: Original image
    
    Returns: 
    statistics of the original image
    """

    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def color_transfer(source, target):
    """
    Arguments: 
    source:   Source image
    target:   Target image
    
    Returns:
    transfer: Modified image
    """
    
#   Convert the images from the RGB to L*ab* color space, being sure to utilizing the 
#   floating point data type!
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target);
    l -= lMeanTar; a -= aMeanTar; b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar/lStdSrc)*l; a = (aStdTar/aStdSrc)*a; b = (bStdTar/bStdSrc)*b;

    # add in the source mean
    l += lMeanSrc; a += aMeanSrc; b += bMeanSrc;

    # clip the pixel intensities to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255); a = np.clip(a, 0, 255); b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


# In[ ]:


def get_random_val(a, b, c, d):
    
    """
    Arguments:
    a, b, c, d: Integers
    
    Returns:
    A random integer in between a and b or c and d depending on threshold calculated as below
    """
    
    delta_1 = b-a; delta_2 = d-c
    
    if(np.random.rand() < delta_1/(delta_1 + delta_2)):
        return np.random.uniform(a, b)
    else:
        return np.random.uniform(c, d)


# In[ ]:


def apply_transformation(img, mask, img_backgnd):

    """
    img:         Original image
    mask:        mask of the relevant subject to be extracted from the image
    img_backgnd: background of the image
    
    Returns: 
    img:         Modified version of the image with the random distortions in it!
    """
    
#   Selecting a random integer in the specified range 
    ind = get_random_val(0.6, 0.9, 1.1, 1.5); 
#   Randomly change the brightness of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Brightness(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
#   Selecting a random integer in the specified range
    ind = get_random_val(0.6, 0.9, 1.1, 1.7);
#   Randomly change the color of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Color(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
#   Selecting a random integer in the specified range
    ind = get_random_val(0.6, 0.9, 1.1, 1.7);
#   Randomly change the contrast of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Contrast(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
    return img


# In[ ]:


relevant_cat = ['person']; relevant_cat_count = [0]

def get_img():
    
    """
    Returns: 
    img:  image with the relevant category in it
    anns: annotations of the image
    """
    ind = np.random.randint(0, len(relevant_cat))
    relevant_cat_count[ind] += 1
    
    catIds = coco.getCatIds(catNms = relevant_cat[ind])
    imgIds = coco.getImgIds(catIds = catIds)
    img = coco.loadImgs(imgIds[relevant_cat_count[ind]])[0]
    
    I = io.imread('%s/%s/%s'%(dataDir, dataType, img['file_name']))
    annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = 0, areaRng = [15000, 50000000])
    anns = coco.loadAnns(annIds)
    
    return I, anns

def GetImg_Foregnd_Backgnd(image, anns):
  
    """
    Arguments: 
    image: Original image
    anns:  Annotations of the corresponding image
    
    Returns: 
    Img_Foregnd: image foreground
    Img_Backgnd: image background
    mask:        mask of the image
    ind:         random index used while selecting the one of the subjects
    """
    
    ind = np.random.randint(0, len(anns)) 
    mask = COCO.annToMask(coco, anns[0])
    img_foregnd = image*np.expand_dims(mask, axis = 2)
    img_backgnd = image - img_foregnd
    
    return img_foregnd, img_backgnd, mask, ind


# In[ ]:


dataDir = './'; dataType = 'train2014'; annFile = '%s/annotations/instances_%s.json'%(dataDir, dataType)

coco = COCO(annFile); cats = coco.loadCats(coco.getCatIds())

print("\n"); categories = [cat['name'] for cat in cats]
print ('COCO categories:\n', ' '.join(categories))

print("\n"); supercategories = set([cat['supercategory'] for cat in cats])
print ('COCO supercategories:\n', ' '.join(supercategories))


# In[ ]:


def get_fakeimg():
    
    """
    Arguments:
    
    Returns: None
    Save the original image, mask of the image, and modified version of the image at the specified location
    """
    
    for i in range(0, 50000):
        
        img, ann = get_img()
        if(len(ann) == 0 or len(img.shape) < 3): continue
        img_foregnd, img_backgnd, mask, ind = GetImg_Foregnd_Backgnd(img, ann)
        
#       mod_img = change_temp_img(img, 100*round(np.random.uniform(35, 40)))*np.expand_dims(mask, axis = 2) + img_backgnd
        mod_img = apply_transformation(img, mask, img_backgnd)
        
        file_loc = ["./Natural/", "./Mask/", "./Composite/"]
        for file in file_loc: 
            if not os.path.exists(file): os.makedirs(file);
              
        plt.imsave(fname = './Natural/img_'   + str(i) + '.jpg', arr = img)
        plt.imsave(fname = './Mask/img_' + str(i) + '_mask.jpg', arr = mask)
        plt.imsave(fname = './Composite/img_' + str(i) + '.jpg', arr = mod_img)
    
    return


# In[ ]:


get_fakeimg()
