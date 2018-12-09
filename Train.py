
# coding: utf-8

# In[1]:


# Import relevant libraries
import numpy as np; np.random.seed(42); import tensorflow as tf; tf.set_random_seed(42);
import matplotlib.pyplot as plt; import pylab; import cv2;
import os; from os import listdir; from os.path import isfile, join
from Architecture import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Image specifications
img_height = 128; img_width = 128; img_channels = 3; batch_size = 1; rel_path = '...Path to this Jupyter Notebook...';

# Define some placeholders
with tf.name_scope("Placeholders"):
  
    lr = tf.placeholder(tf.float32, shape = [], name = "Learning_rate");
    train_mode = tf.placeholder(tf.bool, shape = [], name = "BatchNorm_TrainMode")
    input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_channels], name = "Input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_channels], name = "Input_B")
    global_step = tf.Variable(0, name = "global_step", trainable = False)


# In[3]:


# Output of the generator which should ideally belong to target domain B in our case  
with tf.variable_scope("Generator_", reuse = False): 
    fake_img = generator_unet_128(input_A)
    
# Output of the discriminator which is probability of the real image being 1 
with tf.variable_scope("Discriminator_", reuse = False): 
    prob_real_img = discriminator_patch_gan(input_A, input_B)
# Output of the discriminator which is probability of the fake image being 1
with tf.variable_scope("Discriminator_", reuse = True): 
    prob_fake_img = discriminator_patch_gan(input_A, fake_img)


# In[4]:


def LSGAN_loss():
    
    """
    Returns:
    g_loss: Generator_loss [minimizing the squared difference b/w prob_fake_img and 1]
    d_loss: Discriminator_loss [minimizing the squared difference b/w prob_real_img & 1, and b/w prob_fake_img & 0]
    """

    L1_weight = 200; L1_loss = tf.reduce_mean(tf.abs(input_B - fake_img))
    g_loss = 0.5*tf.reduce_mean(tf.squared_difference(prob_fake_img, 1)) + L1_weight*L1_loss
    d_loss = 0.5*tf.reduce_mean(tf.squared_difference(prob_real_img, 1)) + 0.5*tf.reduce_mean(tf.square(prob_fake_img))
    
    return g_loss, d_loss


# In[5]:


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    
    # Initialize an Adam optimizer with default beta2
    optimizer = tf.train.AdamOptimizer(lr, beta1 = 0.5); model_vars = tf.trainable_variables()

    # Seperate the variables corresponding to the discriminator and generator
    d_vars = [var for var in model_vars if 'Discriminator_' in var.name]
    g_vars = [var for var in model_vars if 'Generator_' in var.name]

    g_loss, d_loss = LSGAN_loss();
    # Define different optimizers for Discriminator and Generator
    d_trainer = optimizer.minimize(d_loss, var_list = d_vars)
    g_trainer = optimizer.minimize(g_loss, var_list = g_vars)


# In[6]:


def generate_fake_validation_images(sess, epoch):

    """
    loc:        Path of the validation set
    num_images: Number of the images in the validation set
    """
    
    loc = '...path to Validation set...'; 
    file_loc = [rel_path + loc + s for s in os.listdir(rel_path + loc)]
    val_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(file_loc))
    val_dataset = val_dataset.map(lambda x: tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), [img_height, img_width]), 127.5), 1))

    val_iterator = val_dataset.make_one_shot_iterator(); next_element = val_iterator.get_next()
    
    if not os.path.exists("./Output/Validation/epoch_" + str(epoch) + "/"):
        os.makedirs("./Output/Validation/epoch_" + str(epoch) + "/")
    
    for i in range(0, len(file_loc)):
        
        try:
#           Get the next image
            next_image = sess.run(validation_next_element)    
#           next_image = np.expand_dims(cv2.cvtColor(next_image, cv2.COLOR_RGBA2RGB), axis = 0)

#           Generate a fake image
            fake_gen_img = sess.run(fake_img, feed_dict = {input_A: next_image})
            
#           Save the fake image at the specified location
            plt.imsave("./Output/Validation/epoch_" + str(epoch) + "/img_" + str(i) + "_fake.png",((fake_gen_img[0] + 1)*127.5).astype(np.uint8))
            plt.imsave("./Output/Validation/epoch_" + str(epoch) + "/img_" + str(i) + ".png",((next_image[0] + 1)*127.5).astype(np.uint8))
            
        except tf.errors.OutOfRangeError: break


# In[7]:


# Specific boolean to be set to True  
Train = True; Test = False; Restore_and_train = False


# In[8]:


def in_training_mode(num_epochs = 100, num_iters, log_dir = "./checkpoints/"):
    
    """
    num_epochs: Number of epochs to train
    log_dir:    Path where to save checkpoints
    num_iters:  Number of training iterations in one epoch
    """
    
    print("Training Started")
    for epoch in range(sess.run(global_step), num_epochs):     

        logs_dir = log_dir; d_l = 0; g_l = 0; num_iters = num_iters;
        if not os.path.exists(logs_dir): os.makedirs(logs_dir)
        sess.run(train_iterator.initializer)

        if epoch < 20: curr_lr = 0.0002;
        elif epoch % 20 == 0: curr_lr = curr_lr/2

        for ptr in range(1, num_iters):
            
            try:
#               Get the next element of the dataset  
                a_next_image, b_next_image = sess.run(train_next_element)
    
#               a_next_image = np.expand_dims(cv2.cvtColor(a_next_image, cv2.COLOR_RGBA2RGB), axis = 0)   
#               b_next_image = np.expand_dims(cv2.cvtColor(b_next_image, cv2.COLOR_RGBA2RGB), axis = 0)

#               Run the train step to update the parameters of the discriminator with 4 times the curr_lr
#               NOTE: This is done to avoid running multiple iterations of discriminator step as in the case of WGAN
                _, dis_loss = sess.run([d_trainer, d_loss], feed_dict = {input_A: a_next_image, input_B: b_next_image, lr: curr_lr})   

#               Run the train step to update the parameters of the generator with the curr_lr
                _, gen_loss = sess.run([g_trainer, g_loss], feed_dict = {input_A: a_next_image, input_B: b_next_image, lr: curr_lr})

#               Calculate the d_loss and g_loss
                d_l += dis_loss; g_l += gen_loss

#               Print some statistics
                if(ptr % 10000 == 0):
                    print(str(epoch*num_iters + ptr) + ' iterations completed and losses are:')
                    print('Generator_loss_: ' + str(g_l/ptr)); print('Discriminator_loss_: ' + str(d_l/ptr))

            except tf.errors.OutOfRangeError:
#               Initialize the iterator again
                sess.run(training_iterator.initializer); continue;

#       Generate fake validation images at the end of each epoch only to check the progress of the model
        generate_fake_validation_images(sess, epoch)
        
#       Save the checkpoints
        saver.save(sess, logs_dir + 'composites', global_step = global_step)
#       Increment the global variable
        sess.run(tf.assign(global_step, epoch + 1))


# In[ ]:


if Train:
    
    """
    num_epochs: number of epochs to run
    """

    # Sorting the filename wrt unique indices of each image
    def sort_composite_filename(x): return int(x[4:-4])

    # Create a datset of composites
    comp_dir = '/images/Composite/';
    comp_fileloc = [rel_path + comp_dir + s for s in sorted(os.listdir(rel_path + comp_dir), key = sort_composite_filename)]
    comp_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(comp_fileloc));
    comp_dataset = comp_dataset.map(lambda x: tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), [img_height, img_width]), 127.5), 1))

    # Sorting the filename wrt unique indices of each image
    def sort_natural_filename(x): return int(x[4:-4])

    # Create a datset of natural images
    nat_dir = '/images/Natural/'
    nat_fileloc = [rel_path + nat_dir + s for s in sorted(os.listdir(rel_path + nat_dir), key = sort_natural_filename)]
    nat_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(nat_fileloc));
    nat_dataset = nat_dataset.map(lambda x: tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), [img_height, img_width]), 127.5), 1))

    # Create a final dataset by zipping the above two [in order to ensure that composite and its ground truth are together]
    train_dataset = tf.data.Dataset.zip((comp_dataset, nat_dataset)).shuffle(2000).repeat()
    train_dataset = (train_dataset.batch(batch_size)).prefetch(10);

    # Create an interator over the dataset 
    train_iterator = train_dataset.make_initializable_iterator(); train_next_element = train_iterator.get_next()
    
    # Set the gpu config options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    
    # Initialize the global variables
    sess.run(tf.global_variables_initializer()); saver = tf.train.Saver(max_to_keep = 10)
    
    in_training_mode(100, len(nat_fileloc));


# In[ ]:


if Restore_and_train:
    
    """
    num_epochs: number of epochs to run
    """
    
#   Set the gpu config options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    sess.run(tf.global_variables_initializer())
    
#   Load the model with the latest checkpoints
    saver = tf.train.Saver(max_to_keep = 10)
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/ls_pix2pix/'))
    print ('Loaded checkpoint! Training from last checkpoint started !!')
    
#   Train it again for n number of epochs
    in_training_mode(num_epochs)


# In[ ]:


if Test:
    
#   Set the config options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    sess.run(tf.global_variables_initializer()); saver = tf.train.Saver()
    
#   Load the model with the latest checkpoints
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/ls_pix2pix/'))
    print ('Loaded checkpoint! Generating fake images corresponding to Test Composites!!')
      
#   Evaluate it on test set  
    loc = '...path to Test set...';
    file_loc = [rel_path + loc + s for s in os.listdir(rel_path + loc)]
    test_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(file_loc))
    test_dataset = test_dataset.map(lambda x: tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), [img_height, img_width]), 127.5), 1))

#   Initialize the test iterator
    test_iterator = test_dataset.make_one_shot_iterator(); test_next_element = test_iterator.get_next()
    
#   Make the directory if it doesn't exists
    if not os.path.exists("./Output/Test/"): os.makedirs("./Output/Test/")
    
#   Set the number of test images in test folder!!
    num_images = 10; 
    for i in range(0, num_images):
        
        try:
#           Get the next element
            next_image = sess.run(test_next_element)
#           next_image = np.expand_dims(cv2.cvtColor(next_image, cv2.COLOR_RGBA2RGB), axis = 0)
            
#           Generate the fake image
            fake_gen_img = sess.run(fake_img, feed_dict = {input_A: next_image})
            
#           Save the image at specified location
            plt.imsave("./Output/Test/fake_img_" + str(i) + ".png",((fake_gen_img[0] + 1)*127.5).astype(np.uint8))
            plt.imsave("./Output/Test/img_" + str(i) + ".png",((next_image[0] + 1)*127.5).astype(np.uint8))
            
        except tf.errors.OutOfRangeError: break

