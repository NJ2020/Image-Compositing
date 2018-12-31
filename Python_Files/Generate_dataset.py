
# coding: utf-8

# In[ ]:


import numpy as np; np.random.seed(42); import skimage; import skimage.io; import skimage.io as io;
import matplotlib.pyplot as plt; from PIL import Image, ImageEnhance; from skimage import img_as_ubyte;
import scipy.optimize; import cv2; import os; from pycocotools.coco import COCO; 
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


# In[ ]:


def get_random_val(a, b, c, d):
    
    """
    Arguments:
    a, b, c, d: Integers
    
    Returns:
    A random integer in between a and b or c and d depending on threshold, calculated as below
    """
    
    delta_1 = b-a; delta_2 = d-c;
    
    if(np.random.rand() < delta_1/(delta_1 + delta_2)):
        return np.random.uniform(a, b)
    else:
        return np.random.uniform(c, d)
    
    
def apply_transformation_1(img, mask, img_backgnd):

    """
    Randomly perturb the contrast, brightness and color of a specific image!
    
    img:         Original image
    mask:        mask of the relevant subject to be extracted from the image
    img_backgnd: background of the image
    
    Returns: 
    img:         Modified version of the image with the random distortions!
    """
    
#   Selecting a random integer in the specified range 
    ind = get_random_val(0.6, 0.9, 1.1, 1.5); 
#   Randomly change the brightness of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Brightness(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
    ind = get_random_val(0.6, 0.9, 1.1, 1.7);
#   Randomly change the color of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Color(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
    ind = get_random_val(0.6, 0.9, 1.1, 1.7);
#   Randomly change the contrast of the image
    img = Image.fromarray(img.astype('uint8'), 'RGB');
    obj = ImageEnhance.Contrast(img); img = obj.enhance(ind);
    img = img*np.expand_dims(mask, axis = 2) + img_backgnd;
    
    return img


# In[ ]:


def pre_transfer_processing(image):

    """
    Arguments:
    image:     Input_image
    
    Returns:
    image_lab: Input_image with the modified L component 
    """
    
#   To effectively stylize images with global transforms, we first compress the dynamic ranges of the two images
#   using a γ (= 2.2) mapping and convert the images into the LAB colorspace

    gamma = 2.2; clip_percent = 0.005;
    
    image = skimage.exposure.adjust_gamma(image, gamma = gamma);
    image_lab = skimage.color.rgb2lab(image); image_lum = image_lab[:, :, 0].copy()

#   Then, we stretch the luminance (L channel) to cover the full dynamic range after clipping both the minimum and
#   the maximum 0.5 percent pixels of luminance levels

    image_lum_flat = image_lum.ravel().copy(); image_lum_sorted = image_lum_flat.argsort()
    lum_cut_pos = int(image_lum_flat.size * clip_percent)
    
    lum_min = image_lum_flat[image_lum_sorted[lum_cut_pos]];
    lum_max = image_lum_flat[image_lum_sorted[image_lum_sorted.size - lum_cut_pos]];
    image_lum[image_lum < lum_min] = lum_min; image_lum[image_lum > lum_max] = lum_max;

    image_lab[:, :, 0] = (image_lum - lum_min) / (lum_max - lum_min) * 100;
    
    return image_lab


# In[ ]:


def ravel_xy_dim(image): return image.reshape(image.shape[0] * image.shape[1], 3);

def get_linear_transformation(sigma_img, sigma_smp):
    
    """
    Arguments:
    sigma_img: Covariance matrix of the original image
    sigma_smp: Covariance matrix of the sample image
    
    Returns:
    linear_transformation_fn: Linear transformation function
    """

#   The solution is unstable for low input covariance values, leading to color artifacts when the input has low 
#   color variation. To avoid this, regularize the solution by clipping diagonal elements of Σ_img as:
#   Σ_img = max(Σ_img , λr*I)

    lambda_r = 7.5; sigma_img = np.maximum(sigma_img, np.eye(sigma_img.shape[0]) * lambda_r);

    [val_i, vec_i] = np.linalg.eig(sigma_img); val_i[val_i < 0] = 0;
    da = np.diag(np.sqrt(val_i + np.finfo(float).eps));
    c = np.matrix(da) * (np.matrix(vec_i).T) * np.matrix(sigma_smp) * np.matrix(vec_i) * np.matrix(da)
    
    [val_c, vec_c] = np.linalg.eig(c); val_c[val_c < 0] = 0;
    dc = np.diag(np.sqrt(val_c + np.finfo(float).eps)); da_inv = np.diag(1 / (np.diag(da)));

#   T = [(Σ_img)**−1/2] * [[(Σ_img)**1/2] * Σ_smp * [(Σ_img)**1/2]]**0.5 * [(Σ_img)**−1/2]

    linear_transformation_fn = (np.matrix(vec_i) * np.matrix(da_inv)) * (np.matrix(vec_c) * np.matrix(dc) *                                (np.matrix(vec_c).T)) * (np.matrix(da_inv) * (np.matrix(vec_i).T))
    
    return linear_transformation_fn
    
def color_transfer_processing(image_original, image_sample):
    
    """
    Arguments:
    image_original: Original image
    image_sample:   Sample image
    
    Returns:
    linear_transformation_fn: Linear transformation function
    """
    
#   Covert the image into 2-dimensional vector (rolling along the spatial dimesnions)
    orig_reshaped = ravel_xy_dim(image_original);  smp_reshaped = ravel_xy_dim(image_sample);
#   Calculate the covariance matrix (3-dimensional)
    sigma_img = np.cov(orig_reshaped.transpose()); sigma_smp = np.cov(smp_reshaped.transpose());
    t = get_linear_transformation(sigma_img, sigma_smp);

#   Both images can be of different size!!
    mu_i = np.repeat([orig_reshaped.mean(axis = 0)], orig_reshaped.shape[0], axis = 0); 
    mu_s = np.repeat([smp_reshaped.mean(axis = 0)],  orig_reshaped.shape[0], axis = 0); 
    xr = (orig_reshaped - mu_i).dot(t) + mu_s

    return np.array(xr).reshape(image_original.shape).copy()


# In[ ]:


def get_lum_transfer_function(lum_image, param):
    
    """
    Arguments:
    lum_image: L channel of the original image
    param:     Parameters to be optimized
    
    Returns:
    num / den: Transfer function
    """
    
    tmp = np.arctan(param[0] / param[1]); 
    num = tmp + np.arctan((lum_image - param[0]) / param[1]); den = tmp + np.arctan((1 - param[0]) / param[1])

    return  num / den;

def get_lum_cost(param, lum_input, lum_cal):
    
    """
    Arguments:
    lum_input: L channel of the original image
    param:     L calculated
    
    Returns:
    num / den: squared L2 norm of the || g(lum_input) − lum_cal ||, g is transfer function
    """
    return np.power(np.linalg.norm(get_lum_transfer_function(lum_input, param) - lum_cal, 2), 2);

def extract_lum_feature(image, bins = 100, num_of_samples = 32):
    
    """
    Arguments:
    image: Input image
    bins:  Total bins in which to categorize the pixel intensities
    num_of_samples: # uniformly sampled percentiles of the luminance cumulative distribution function
    
    Returns:
    bins[index]: Luminance values at the percentiles calculated from above.
    """
    
    hist, bins = np.histogram(image.ravel(), bins = bins, normed = True); cdf = np.cumsum(hist); 
    percents = np.arange(1, 1 + num_of_samples) / num_of_samples; index = np.searchsorted(cdf, percents);
    
    return bins[index]

def luminance_transfer_processing(image_original, image_sample, image_result):
    
    """
    Arguments:
    image_original: Original image
    image_sample:   Sample image
    image_result:   Final image
    
    Returns:
    image_result:   Final image
    """
    
    tau = 0.4; num_of_samples = 32;

    lum_image  = extract_lum_feature(image_original[:, :, 0], num_of_samples = num_of_samples)
    lum_sample = extract_lum_feature(image_sample[:, :, 0],   num_of_samples = num_of_samples)
    lum_cal = lum_image + (lum_sample - lum_image) * (tau / np.minimum(tau, np.linalg.norm(lum_sample - lum_image, np.inf)))

    target_function = lambda para: get_lum_cost(para, lum_image, lum_cal)
    result = scipy.optimize.minimize(target_function, np.random.random_sample([2]));
    image_result[:, :, 0] = get_lum_transfer_function(image_result[:, :, 0], result.x)
    
    return image_result


# In[ ]:


def apply_transformation_2(img_1, mask_1, img_1_backgnd, img_2, mask_2, img_2_backgnd):
    
    """
    img_1:         Original image
    mask_1:        mask of the relevant subject to be extracted from the image
    img_backgnd_1: background of the image
    
    Returns: 
    final_image:   Perturbed version of the image with the random distortions!
    """
    
    luminance_transfer = False;
    image_original = pre_transfer_processing(skimage.img_as_float(img_1));
    image_sample   = pre_transfer_processing(skimage.img_as_float(img_2));

    final_image = color_transfer_processing(image_original, image_sample);
    if luminance_transfer: final_image = luminance_transfer_processing(image_original, image_sample, final_image);
        
    final_image = skimage.color.lab2rgb(final_image); gamma = 1 / 2.2; 
    final_image = skimage.exposure.adjust_gamma(final_image, gamma = gamma); final_image = img_as_ubyte(final_image); 
    final_image = final_image*np.expand_dims(mask_1, axis = 2) + img_1_backgnd;
    
    return final_image


# In[ ]:


relevant_cat = ['person']; relevant_cat_img_count = [0]*len(relevant_cat);

def get_img(bool_orig_img = True):
    
    """
    Returns: 
    img:  image with the relevant category in it
    anns: annotations of the image
    """
    
    ind = np.random.randint(0, len(relevant_cat));
    catIds = coco.getCatIds(catNms = relevant_cat[ind]); imgIds = coco.getImgIds(catIds = catIds);
    
    if bool_orig_img: relevant_cat_img_count[ind] += 1; img = coco.loadImgs(imgIds[relevant_cat_img_count[ind]])[0];
    else: img = coco.loadImgs(imgIds[np.random.randint(low = 0, high = len(imgIds))])[0];

    I = io.imread('%s/%s/%s'%(dataDir, dataType, img['file_name']))
    annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = 0, areaRng = [15000, 50000000])
    anns = coco.loadAnns(annIds);

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
    
    ind = np.random.randint(0, len(anns)) ; mask = COCO.annToMask(coco, anns[0])
    img_foregnd = image*np.expand_dims(mask, axis = 2); img_backgnd = image - img_foregnd
    
    return img_foregnd, img_backgnd, mask, ind


# In[ ]:


def get_fakeimg():
    
    """
    Arguments:
    
    Returns: None
    Save the original image, mask of the image, and modified version of the image at the specified location
    """
    
    for i in range(0, 50000):
        
        while(True):
            img_1, ann_1 = get_img(bool_orig_img = True);
            if(len(ann_1) == 0 or len(img_1.shape) < 3): continue
            else: break;
        
        while(True):
            img_2, ann_2 = get_img(bool_orig_img = False);
            if(len(ann_2) == 0 or len(img_2.shape) < 3): continue
            else: break;
                
        img_1_foregnd, img_1_backgnd, mask_1, ind_1 = GetImg_Foregnd_Backgnd(img_1, ann_1);
        img_2_foregnd, img_2_backgnd, mask_2, ind_2 = GetImg_Foregnd_Backgnd(img_2, ann_2)
        
        mod_img_1 = apply_transformation_1(img_1, mask_1, img_1_backgnd);
        mod_img_2 = apply_transformation_2(img_1, mask_1, img_1_backgnd, img_2, mask_2, img_2_backgnd);
        
        file_loc = ["./Natural/", "./Mask/", "./Composite_1/", "./Composite_2/"]
        for file in file_loc: 
            if not os.path.exists(file): os.makedirs(file);
              
        plt.imsave(fname = './Natural/img_'   + str(i) + '.jpg', arr = img_1)
        plt.imsave(fname = './Mask/img_' + str(i) + '_mask.jpg', arr = mask_1)
        plt.imsave(fname = './Composite_1/img_' + str(i) + '.jpg', arr = mod_img_1)
        plt.imsave(fname = './Composite_2/img_' + str(i) + '.jpg', arr = mod_img_2)
        
    return


# In[ ]:


dataDir = './'; dataType = 'train2014'; annFile = '%s/annotations/instances_%s.json'%(dataDir, dataType)

coco = COCO(annFile); cats = coco.loadCats(coco.getCatIds())

print("\n"); categories = [cat['name'] for cat in cats]
print ('COCO categories:\n', ' '.join(categories))

print("\n"); supercategories = set([cat['supercategory'] for cat in cats])
print ('COCO supercategories:\n', ' '.join(supercategories))


# In[ ]:


get_fakeimg()

