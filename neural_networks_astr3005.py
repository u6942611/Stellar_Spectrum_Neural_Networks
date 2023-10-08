
from __future__ import absolute_import, division, print_function # python2 compatibility
import torch
from The_Payne import utils
from The_Payne import spectral_model
from The_Payne import fitting

import numpy as np
import sys
import os
import torch
import time
from torch.autograd import Variable
from The_Payne import radam
try:
    # try to improve resolution for apple
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

# basic packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# pickle is one of the many formats that can be used to store large files
import pickle

# packages to monitor resources
import psutil
import time
import csv
import pandas as pd
#!/usr/bin/env python
# coding: utf-8

# # Testing neural network interpolation for stellar spectra
# 
# ### Basics:
# 
# In preparation for this project, the stellar spectrum synthesis program Spectroscopy Made Easy (`sme`) was used to create 1960 spectra with a total of 64224 pixels at high resolution (R = 300000) for the 4 CCDs of the HERMES spectrograph that is used for the GALAH survey.
# 
# The necessary information for these 1960 spectra comes in different files:
# 
# 1 `wavelength` to map px -> wavelenght; wavelength.shape (1, 64224)  
# 2 `flux` with flux.shape (1960, 64224)  
# 3 `labels` with stellar labels (temperature, iron abundance etc.)  

# In[ ]:


save_dir = 'av_results_full'    #Name whatever your directory in AVATAR is called. This is where all the results are going. 



# In[ ]:





# In[2]:


# Preamble



# In[3]:


# Sleep before saving the initial resources
time.sleep(5)


# In[4]:


# Create empty arrays that will be populated during each monitoring step
used_time = []
used_cpu_percent = []
used_loadavg = []
used_vmem_gb = []




# Read in the wavelength array that will help us to connext 64224 pixels with wavelengths in Aangstroem.
file = open('data/galah_dr4_3dbin_wavelength_array.pickle','rb')
wavelength = pickle.load(file)
file.close()



 
wavelength = wavelength#[:int(16000)]     #Reading full wavelength




# Read in the flux array for 1960 spectra with 64224 pixles, i.e. flux.shape (1960, 64224)
file = open('data/galah_dr4_trainingset_5750_4.50_0.00_incl_vsini_flux_ivar.pickle','rb')
flux = pickle.load(file)
file.close()





# Read in the stellar labels table with labels == 'teff','logg','fe_h','vmic','vsini','li_fe', etc.
labels = Table.read('data/galah_dr4_trainingset_5750_4.50_0.00_incl_vsini.fits')

desired_labels = [ 'teff', 'logg', 'fe_h', 'vmic', 'vsini', 'li_fe', 'c_fe', 'n_fe', 'o_fe', 
                  'na_fe', 'mg_fe', 'al_fe',  'si_fe', 'k_fe', 'ca_fe', 'sc_fe', 'ti_fe', 'v_fe', 'cr_fe', 'mn_fe', 'co_fe', 
                  'ni_fe', 'cu_fe', 'zn_fe',  'rb_fe', 'sr_fe', 'y_fe', 'zr_fe', 'mo_fe', 'ru_fe', 'ba_fe', 'la_fe', 
                  'ce_fe', 'nd_fe', 'sm_fe', 'eu_fe']


labels = ([labels[key] for key in desired_labels])
galah_labels = np.vstack(labels).T


#Splitting the dataset  70/15/15

np.random.seed(42)
indices = np.arange(len(galah_labels))
np.random.shuffle(indices)

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the split points
train_split = int(train_ratio * len(galah_labels))
val_split = int((train_ratio + val_ratio) * len(galah_labels))

# Split the labels
galah_labels_train = galah_labels[indices[:train_split]]
galah_labels_val = galah_labels[indices[train_split:val_split]]
galah_labels_test = galah_labels[indices[val_split:]]

flux = flux#[:, :16000]
galah_spectra_train = flux[indices[:train_split]]
galah_spectra_val = flux[indices[train_split:val_split]]
galah_spectra_test = flux[indices[val_split:]]




def plot_and_save_spectrum(index, show = True):
    """
    Plot the spectrum for a given index of the flux array

    INPUT:
    index : int
    """
    f, gs = plt.subplots(4,1,figsize=(10,6),sharey=True)

    # We have 4 CCDs that we loop over
    for ccd in [1,2,3,4]:
        
        ax = gs[ccd-1]
        
        # Select all pixels of the spectrum that are in that CCD range
        # They roughly come in bins of 4000-5000, 5000-6000, 6000-7000, and 7000-8000
        pixel_in_ccd = (wavelength > (3 + ccd)*1000) & (wavelength < (4 + ccd)*1000)

        # Plot the spectrum by selecting the right index of the flux array
        ax.plot(
            wavelength[pixel_in_ccd],
            flux[index, pixel_in_ccd],
            lw = 1
        )

        ax.set_ylabel('Flux / norm.',fontsize=15)
        if ccd == 4:
            ax.set_xlabel('Wavelength / $\mathrm{\AA}$',fontsize=15)

    # Get rid of white space
    plt.tight_layout(w_pad=0,h_pad=0)
    
    # Save into directory figures
    plt.savefig('figures/input_spectrum_index_'+str(index)+'.pdf',bbox_inches='tight',dpi=200)
    if show:
        plt.show()
    plt.close()


#Defining the functions
    
# simple multi-layer perceptron model
class Payne_Relu(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_Relu, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)
    
    



class Payne_Sigmoid(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_Sigmoid, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)
    
# payne mix - one sigmoid one relu

class Payne_Mix(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_Mix, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)
    
    
class Payne_Mix2(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_Mix2, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)
    
    
class Payne_ELU(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_ELU, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.ELU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.ELU(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)

    
class Payne_Tanh(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_Tanh, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.Tanh(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.Tanh(),
            torch.nn.Linear(num_neurons, num_pixel),
        )

    def forward(self, x):
        return self.features(x)




# In[17]:


'''
This file is used to train the neural net that predicts the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is slightly different from Ting+19. Instead of
training individual small networks for each pixel separately, here we train a single
large network for all pixels simultaneously.

The advantage of doing so is that individual pixels will exploit information
from adjacent pixels. This usually leads to more precise interpolations.

However to train a large network, GPU is needed. This code will
only run with GPU. But even with an inexpensive GPU, this code
should be pretty efficient -- training with a grid of 10,000 training spectra,
with > 10 labels, should not take more than a few hours

The default training set are the Kurucz synthetic spectral models and have been
convolved to the appropriate R (~22500 for APOGEE) with the APOGEE LSF.
'''


'''
class Payne_model(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_features),
        )

        self.deconv1 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv2 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv3 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv5 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv6 = torch.nn.ConvTranspose1d(64, 32, mask_size, stride=3, padding=5)
        self.deconv7 = torch.nn.ConvTranspose1d(32, 1, mask_size, stride=3, padding=5)

        self.deconv2b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv3b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv4b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv5b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv6b = torch.nn.ConvTranspose1d(64, 32, 1, stride=3)

        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU()
        self.relu4 = torch.nn.LeakyReLU()
        self.relu5 = torch.nn.LeakyReLU()
        self.relu6 = torch.nn.LeakyReLU()

        self.num_pixel = num_pixel

    def forward(self, x):
        x = self.features(x)[:,None,:]
        x = x.view(x.shape[0], 64, 5)
        x1 = self.deconv1(x)

        x2 = self.deconv2(x1)
        x2 += self.deconv2b(x1)
        x2 = self.relu2(x2)

        x3 = self.deconv3(x2)
        x3 += self.deconv3b(x2)
        x3 = self.relu2(x3)

        x4 = self.deconv4(x3)
        x4 += self.deconv4b(x3)
        x4 = self.relu2(x4)

        x5 = self.deconv5(x4)
        x5 += self.deconv5b(x4)
        x5 = self.relu2(x5)

        x6 = self.deconv6(x5)
        x6 += self.deconv6b(x5)
        x6 = self.relu2(x6)

        x7 = self.deconv7(x6)[:,0,:self.num_pixel]
        return x7
'''


activation_classes = ["Payne_Relu", "Payne_Sigmoid", "Payne_Mix", "Payne_Mix2"]




#===================================================================================================
# train neural networks
def neural_net(training_labels, training_spectra, validation_labels, validation_spectra,  act_function, cuda,\
             num_neurons = 300, num_steps=1e4, learning_rate=1e-4, batch_size=512,\
             num_features = 64*5, mask_size=11, num_pixel=64224):

    '''
    Training a neural net to emulate spectral models

    training_labels has the dimension of [# training spectra, # stellar labels]
    training_spectra has the dimension of [# training spectra, # wavelength pixels]

    The validation set is used to independently evaluate how well the neural net
    is emulating the spectra. If the neural network overfits the spectral variation, while
    the loss will continue to improve for the training set, but the validation
    set should show a worsen loss.

    The training is designed in a way that it always returns the best neural net
    before the network starts to overfit (gauged by the validation set).

    num_steps = how many steps to train until convergence.
    1e4 is good for the specific NN architecture and learning I used by default.
    Bigger networks will take more steps to converge, and decreasing the learning rate
    will also change this. You can get a sense of how many steps are needed for a new
    NN architecture by plotting the loss evaluated on both the training set and
    a validation set as a function of step number. It should plateau once the NN
    has converged.

    learning_rate = step size to take for gradient descent
    This is also tunable, but 1e-4 seems to work well for most use cases. Again,
    diagnose with a validation set if you change this.

    num_features is the number of features before the deconvolutional layers; it only
    applies if ResNet is used. For the simple multi-layer perceptron model, this parameter
    is not used. We truncate the predicted model if the output number of pixels is
    larger than what is needed. In the current default model, the output is ~8500 pixels
    in the case where the number of pixels is > 8500, increase the number of features, and
    tweak the ResNet model accordingly

    batch_size = the batch size for training the neural networks during the stochastic
    gradient descent. A larger batch_size reduces stochasticity, but it might also
    risk of stucking in local minima

    '''

    # run on cuda
    #dtype = torch.cuda.FloatTensor
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    if torch.cuda.is_available() and cuda == True:
     dtype = torch.cuda.FloatTensor
     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
     dtype = torch.FloatTensor
     torch.set_default_tensor_type('torch.FloatTensor')

    # scale the labels, optimizing neural networks is easier if the labels are more normalized
    x_max = np.max(training_labels, axis = 0)
    x_min = np.min(training_labels, axis = 0)
    x = (training_labels - x_min)/(x_max - x_min) - 0.5
    x_valid = (validation_labels-x_min)/(x_max-x_min) - 0.5

    # dimension of the input
    dim_in = x.shape[1]

#--------------------------------------------------------------------------------------------
    # assume L1 loss
    loss_fn = torch.nn.L1Loss(reduction = 'mean')

    # make pytorch variables
    x = Variable(torch.from_numpy(x)).type(dtype)
    y = Variable(torch.from_numpy(training_spectra), requires_grad=False).type(dtype)
    x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
    y_valid = Variable(torch.from_numpy(validation_spectra), requires_grad=False).type(dtype)

    
    print(act_function)
    
    
    payne_model_function_name = globals().get(act_function)
    
    print(type(payne_model_function_name))
    
    
    # initiate Payne and optimizer
    model = payne_model_function_name(dim_in, num_neurons, num_features, mask_size, num_pixel)
    
    '''
    this is what i want to be adjusting it seems ^^^^^^^^^^^^^^^^^^
    
    '''  
    
    
    
    
    if torch.cuda.is_available() and cuda == True:
        model.cuda()
    model.train()

    # we adopt rectified Adam for the optimization
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

#--------------------------------------------------------------------------------------------
    # train in batches
    nsamples = x.shape[0]
    nbatches = nsamples // batch_size

    nsamples_valid = x_valid.shape[0]
    nbatches_valid = nsamples_valid // batch_size

    # initiate counter
    current_loss = np.inf
    training_loss =[]
    validation_loss = []

#-------------------------------------------------------------------------------------------------------
    # train the network
    for e in range(int(num_steps)):

        # randomly permute the data
        perm = torch.randperm(nsamples)
        if torch.cuda.is_available() and cuda == True:
            perm = perm.cuda()

        # for each batch, calculate the gradient with respect to the loss
        for i in range(nbatches):
            idx = perm[i * batch_size : (i+1) * batch_size]
            y_pred = model(x[idx])

            loss = loss_fn(y_pred, y[idx])*1e4
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

#-------------------------------------------------------------------------------------------------------
        # evaluate validation loss
        if e % 100 == 0:

            # here we also break into batches because when training ResNet
            # evaluating the whole validation set could go beyond the GPU memory
            # if needed, this part can be simplified to reduce overhead
            perm_valid = torch.randperm(nsamples_valid)
            if torch.cuda.is_available():
                perm_valid = perm_valid.cuda()
            loss_valid = 0

            for j in range(nbatches_valid):
                idx = perm_valid[j * batch_size : (j+1) * batch_size]
                y_pred_valid = model(x_valid[idx])
                loss_valid += loss_fn(y_pred_valid, y_valid[idx])*1e4
            loss_valid /= nbatches_valid

            print('iter %s:' % e, 'training loss = %.3f' % loss,\
                 'validation loss = %.3f' % loss_valid)

            loss_data = loss.detach().data.item()
            loss_valid_data = loss_valid.detach().data.item()
            training_loss.append(loss_data)
            validation_loss.append(loss_valid_data)

#--------------------------------------------------------------------------------------------
            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid_data
                model_numpy = []
                for param in model.parameters():
                    model_numpy.append(param.data.cpu().numpy())

                # extract the weights and biases
                w_array_0 = model_numpy[0]
                b_array_0 = model_numpy[1]
                w_array_1 = model_numpy[2]
                b_array_1 = model_numpy[3]
                w_array_2 = model_numpy[4]
                b_array_2 = model_numpy[5]
                
                

                # save parameters and remember how we scaled the labels
                np.savez(f"{save_dir}/NN_normalized_spectra_{act_function}_{num_neurons}.npz",\
                        w_array_0 = w_array_0,\
                        w_array_1 = w_array_1,\
                        w_array_2 = w_array_2,\
                        b_array_0 = b_array_0,\
                        b_array_1 = b_array_1,\
                        b_array_2 = b_array_2,\
                        x_max=x_max,\
                        x_min=x_min,)

                # save the training loss
                np.savez(f"{save_dir}/training_loss_{act_function}_{num_neurons}.npz",\
                         training_loss = training_loss,\
                         validation_loss = validation_loss)

#--------------------------------------------------------------------------------------------
    # extract the weights and biases
    w_array_0 = model_numpy[0]
    b_array_0 = model_numpy[1]
    w_array_1 = model_numpy[2]
    b_array_1 = model_numpy[3]
    w_array_2 = model_numpy[4]
    b_array_2 = model_numpy[5]

    # save parameters and remember how we scaled the labels
    np.savez(f"{save_dir}/NN_normalized_spectra_{act_function}_{num_neurons}.npz",\
             w_array_0 = w_array_0,\
             w_array_1 = w_array_1,\
             w_array_2 = w_array_2,\
             b_array_0 = b_array_0,\
             b_array_1 = b_array_1,\
             b_array_2 = b_array_2,\
             x_max=x_max,\
             x_min=x_min,)

    # save the final training loss
    np.savez(f"{save_dir}/training_loss_{act_function}_{num_neurons}.npz",\
             training_loss = training_loss,\
             validation_loss = validation_loss)

    return



# label array unit = [n_spectra, n_labels]
# spectra_array unit = [n_spectra, n_pixels]

# The validation set is used to independently evaluate how well the neural net
# is emulating the spectra. If the network overfits the spectral variation, while 
# the loss will continue to improve for the training set, the validation set
# should exhibit a worsen loss.

# the codes outputs a numpy array ""NN_normalized_spectra.npz" 
# which stores the trained network parameters
# and can be used to substitute the default one in the directory neural_nets/
# it will also output a numpy array "training_loss.npz"
# which stores the progression of the training and validation losses







def get_mse(index, model):
    #model is the .npz coefficients file
    
    x_min = model[6]
    x_max = model[7]
    testing_spectra = galah_spectra_test[index]
    real_labels =scaled_labels= galah_labels_test[index]

    scaled_labels = (real_labels-x_min)/(x_max-x_min) - 0.5
    model_spec = spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = model)
    
    mse = np.mean((model_spec - testing_spectra) ** 2)
    
    return mse




#Model is the NNcoefs that have been derived from the specific neural net
def get_square_dif_per_pix(index, model):
    real_labels =scaled_labels= galah_labels_test[index]
    testing_spectra = galah_spectra_test[index]
    x_min = model[6]
    x_max = model[7]
    scaled_labels = (real_labels-x_min)/(x_max-x_min) - 0.5
    model_spec = spectral_model.get_spectrum_from_neural_net(scaled_labels = scaled_labels, NN_coeffs = model)

    error = (model_spec-testing_spectra)**2
    
    return error
    


def get_mae(model):
    '''
    median approximation error
    '''
    nested_mae_array = []
    mae_final = []
    for a in np.arange(0,len(galah_spectra_test)):
        square_difs = get_square_dif_per_pix(a, model)
        nested_mae_array.append(square_difs)
    
    mae_final = np.median(nested_mae_array, axis=0)
    return mae_final
        
        

#Function to train each variation of the neural network and save data in folder.
# This assumes my directory in AVATAR is called "av_results_full" 


def train_model(num_neurons, act_function):
    '''
    Will return time arrays, avg mse, rmse, mae_array
    '''
    print(type(act_function))
    start_time = time.time()
    start_used_cpu_percent = psutil.cpu_percent()
    start_loadavg_1_5_15 = np.array(psutil.getloadavg())
    start_used_vmem_gb = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    used_time = []
    used_cpu_percent = []
    used_loadavg = []
    used_vmem_gb = []
    
    #TRAINING
    neural_net(galah_labels_train, galah_spectra_train,\
                    galah_labels_val, galah_spectra_val, act_function, cuda = True,
                    num_neurons=num_neurons, learning_rate=1e-4,\
                    num_steps=2e3, batch_size=128)
    
    used_time.append(time.time() - start_time)

    used_cpu_percent.append(psutil.cpu_percent() - start_used_cpu_percent)
    used_loadavg.append(np.array(psutil.getloadavg()) - start_loadavg_1_5_15)
    used_vmem_gb.append(psutil.virtual_memory().used / (1024 * 1024 * 1024) - start_used_vmem_gb)
    
    #The following will exist after the above training step is completed.

    tmp = np.load(f"av_results_full/NN_normalized_spectra_{act_function}_{num_neurons}.npz")
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    tmp.close()
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    
    
    #Use loaded coefficients to get metrics
    
    mae = get_mae(NN_coeffs) #Median Approximation Error
    
    #RMSE and AVG MSE
    
    mse_array = []
    
    for i in np.arange(len(galah_spectra_test)):
        mse_array.append(get_mse(i,NN_coeffs))
        
    avg_mse = np.round(np.sum(mse_array)/len(galah_spectra_test),5)
    rmse = np.round(np.sqrt(np.average(mse_array)),5)
    
    
    
    return [used_time, used_cpu_percent, used_loadavg, used_vmem_gb], avg_mse, rmse, mae
    
    


activation_classes = ["Payne_Relu", "Payne_Sigmoid", "Payne_Mix", "Payne_Mix2", "Payne_ELU", "Payne_Tanh"]
num_neurons = [20, 50, 100, 200, 300, 400, 500]

#Implement the function and run through the possible functions and neurons.

for n in num_neurons:
    for a in activation_classes:
        print("doing", a, "with", n)
        training_results = train_model( n ,a)
        dic_stats = {'Num Neurons': n, 'Time Taken': training_results[0][0], 'CPU Percent': training_results[0][1],
                             'vmem_gb': training_results[0][3],
                            'AVG MSE': training_results[1] , 'RMSE': training_results[2]}
        dataframe_stats = pd.DataFrame(dic_stats)
        stats_file_name = f"{n}_neurons_{a}_metrics.csv"
        dataframe_stats.to_csv(f"{save_dir}/{stats_file_name}")    
        rmse_frame = pd.DataFrame({f"{a}_with_{n}_neurons MAE": training_results[3]})


        rmse_file_name = f"{n}_neurons_{a}_mae.csv"
        rmse_frame.to_csv(f"{save_dir}/{rmse_file_name}")

