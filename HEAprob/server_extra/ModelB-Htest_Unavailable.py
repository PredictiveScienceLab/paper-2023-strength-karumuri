#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import GPy
import pymc3 as pm
from warnings import filterwarnings
filterwarnings('ignore')
# import seaborn as sns
# sns.set() # Setting seaborn as default style even if use only matplotlib
import argparse
import sys
import os
os.environ['PYTHONHASHSEED'] = '0'
plt.rcParams.update({'font.size': 9})


# In[2]:
#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-seed', dest = 'seed', type = int, 
                    default = 48, help  = 'Seed to split train and test data')
args = parser.parse_args()

seed = seed_subset_selection = args.seed # Randomness 
np.random.seed(seed)


# In[3]:


resultdir = os.path.join(os.getcwd(), 'ModelB-Htest_Unavailable_results' ,str(seed)+'-seed') 

if not os.path.exists(resultdir):
    os.makedirs(resultdir)

orig_stdout = sys.stdout
q = open(os.path.join(resultdir, str(seed)+'-output'+'.txt'), 'w')
sys.stdout = q


# In[4]:


import os
path = (os.path.join(os.getcwd(), '..'))


# ### NSF DMREF - Experimental

# Consider the following dataset:

# In[5]:


# this way of doing just to make sure that test data is same across ModelA and ModelB for comparison

raw_data_Honly = pd.read_csv(os.path.join(path,'Hardness_only_with_Descriptors-Sh.csv'), index_col=0)
raw_data_Sonly = pd.read_csv(os.path.join(path,'Strength_only_with_Descriptors-Sh.csv'), index_col=0)
raw_data = pd.read_csv(os.path.join(path,'Strength_Hardness_with_Descriptors-Sh.csv'), index_col=0)

# raw_data_Honly = pd.read_csv('Hardness_only_with_Descriptors-Sh.csv')
# raw_data_Sonly = pd.read_csv('Strength_only_with_Descriptors-Sh.csv')
# raw_data = pd.read_csv('Strength_Hardness_with_Descriptors-Sh.csv')

# display(raw_data_Honly)
# display(raw_data_Sonly)
# display(raw_data)


# In[6]:


print (raw_data_Honly.shape[0]+raw_data.shape[0], raw_data_Sonly.shape[0]+raw_data.shape[0])


# Extracting all relevant features:

# In[7]:


# The unscaled input and output features
RoM = ['Density', 'Modulus', 'Tm', 'Misfit', 'Rbar', 'delta', 'shear', 'bulk', 'VEC', 'SMIX']
MX = ['ModulusMX', 'DensityMX', 'TmMX', 'R_MX', 'shearMX', 'bulkMX', 'VECMX']
Phase = ['Phase_BCC', 'Phase_BCC+BCC', 'Phase_BCC+Sec.', 'Phase_FCC', 'Phase_FCC+BCC', 'Phase_FCC+Sec.', 'Phase_No Data', 'Phase_Other']
X = RoM + MX + Phase
# Curtin = 'Delta_SS'
H = 'PROPERTY: HV'
S = 'PROPERTY: YS (MPa)'


# # Model B

# In this Case, we are going to build a Strength model using MFGP by building two GPs

# We build GP1 using {X=X1(below), Y=H}, mean prediction = H_pred
# 
# We build GP2 using {X=[X2(below)+H_pred(X2)], Y=S}, mean prediction = S_pred

# In[8]:


print ('Splitting')
from sklearn.model_selection import train_test_split
raw_data_train, raw_data_test = train_test_split(raw_data, test_size=0.2, random_state=seed_subset_selection)
print (raw_data_train.shape, raw_data_test.shape)
# print (raw_data_train[:2])
# print (raw_data_test[:2])


# In[9]:


print ('Pre-processing the data for MFGP - GP1 and GP2 building')
print ('Step1: Seperating')
X1_train = np.vstack(( raw_data_Honly[X], raw_data_train[X] ))
y1_train = np.vstack(( raw_data_Honly[H].values[:, None], raw_data_train[H].values[:, None] ))
X2_train = raw_data_train[X]
y2_train = raw_data_train[S].values[:, None] 
X_test, y1_test, y2_test = raw_data_test[X].values, raw_data_test[H].values[:, None], raw_data_test[S].values[:, None] 

print('Step2: Scaling [X1, X2, H], Scaling S')
# scale train data and use those means and dev for scaling test data to avoid info leakage
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler() # using single scaler_X, for X's in all fidelities.
scaler_X.fit(X2_train)
X1_train_s = scaler_X.transform(X1_train)
X2_train_s = scaler_X.transform(X2_train)
X_test_s = scaler_X.transform(X_test) # X_test_s = X1_test_s = X2_test_s
scaler_y1 = StandardScaler()
scaler_y1.fit(y1_train)
y1_train_s, y1_test_s = scaler_y1.transform(y1_train), scaler_y1.transform(y1_test)
scaler_y2 = StandardScaler()
scaler_y2.fit(y2_train)
y2_train_s, y2_test_s = scaler_y2.transform(y2_train), scaler_y2.transform(y2_test)


# In[10]:


print ('Training and testing dataset sizes for GP1')
print (np.shape(X1_train_s), np.shape(y1_train_s))
print (np.shape(X_test_s), np.shape(y1_test_s))

print ('Training and testing dataset sizes for GP2')
print (np.shape(X2_train_s), np.shape(y2_train_s))
print (np.shape(X_test_s), np.shape(y2_test_s))


# In[11]:


print ('Building GP1 between X1_s and H_s')
import GPy
k1 = GPy.kern.RBF(X1_train_s.shape[1], ARD=True)
m1 = GPy.models.GPRegression(X1_train_s, y1_train_s, k1)
m1.optimize_restarts()
print(m1)
print(m1[".*Gaussian_noise"])
print(m1[".*rbf"])

# Evaluate the previous fidelity model on the inputs of this fidelity
f1, _ = m1.predict(X2_train_s) # H_pred_s

print ('Building GP2 between [X2_s+H_pred_s] and S_s')
import GPy
k2 = GPy.kern.RBF(np.hstack((X2_train_s,f1)).shape[1], ARD=True)
m2 = GPy.models.GPRegression(np.hstack((X2_train_s,f1)), y2_train_s, k2)
m2.optimize_restarts()
print(m2)
print(m2[".*Gaussian_noise"])
print(m2[".*rbf"])


# In[12]:


def posterior_samples_f(X, size=100):
    # To store the samples
    Ys = []
    F1 = m1.posterior_samples_f(X[:,:m1.X.shape[1]], size=size) # The _f is required
    Ys.append(F1)
    # To store samples from this fidelity
    Ysi = np.ndarray((X.shape[0], 1, size))
    for s in range(size):
        Xsi =  np.hstack(( X, Ys[0][:, :, s] ))
        # This is the computationally expensive step:
        Ysi[:, :, s] = m2.posterior_samples_f(Xsi, size=1)[:, :, 0]
    Ys.append(Ysi)
    return Ys


# In[13]:


print('MFGP Predictions: Hardness')  # posterior (f1|y and f1*|y)
postsam1_train = posterior_samples_f(X1_train_s, size=1000)[0] # posterior_samples_f gives epistemic uncertainity only
postsam1_train_un = scaler_y1.inverse_transform(postsam1_train) # un-normalized values 
print (postsam1_train_un.shape)

postsam1_test = posterior_samples_f(X_test_s, size=1000)[0] # posterior_samples_f gives epistemic uncertainity only
postsam1_test_un = scaler_y1.inverse_transform(postsam1_test) # un-normalized values
print (postsam1_test_un.shape)

mupred1_train, varpred1_train = np.mean(postsam1_train_un, axis=2), np.var(postsam1_train_un, axis=2)
mupred1_test, varpred1_test = np.mean(postsam1_test_un, axis=2), np.var(postsam1_test_un, axis=2)

errbarpred1_train = 1.96*np.sqrt(varpred1_train)
errbarpred1_test = 1.96*np.sqrt(varpred1_test)

print('MFGP Predictions: Strength')  # posterior (f2|y and f2*|y)
postsam2_train = posterior_samples_f(X2_train_s, size=1000)[1] # posterior_samples_f gives epistemic uncertainity only
postsam2_train_un = scaler_y2.inverse_transform(postsam2_train) # un-normalized values 
print (postsam2_train_un.shape)

postsam2_test = posterior_samples_f(X_test_s, size=1000)[1] # posterior_samples_f gives epistemic uncertainity only
postsam2_test_un = scaler_y2.inverse_transform(postsam2_test) # un-normalized values
print (postsam2_test_un.shape)

mupred2_train, varpred2_train = np.mean(postsam2_train_un, axis=2), np.var(postsam2_train_un, axis=2)
mupred2_test, varpred2_test = np.mean(postsam2_test_un, axis=2), np.var(postsam2_test_un, axis=2)

errbarpred2_train = 1.96*np.sqrt(varpred2_train)
errbarpred2_test = 1.96*np.sqrt(varpred2_test)


# In[14]:


from sklearn.metrics import mean_absolute_error
def mean_relative_absolute_error(y_true, y_pred): 
    return np.mean( np.abs((y_pred-y_true)/y_true) )

print('MFGP Predictions plots: Hardness')
Hardness_MRelAE_train = mean_relative_absolute_error( y1_train, mupred1_train )
Hardness_MRelAE_test = mean_relative_absolute_error( y1_test, mupred1_test )

Hardness_MAE_train = mean_absolute_error( y1_train,  mupred1_train )
Hardness_MAE_test = mean_absolute_error( y1_test,  mupred1_test )

print ('************************************************')
print ('Hardness MRel.AE train: '+str(Hardness_MRelAE_train))
print ('Hardness MRel.AE test: '+str(Hardness_MRelAE_test))

print ('Hardness MAE train: '+str(Hardness_MAE_train))
print ('Hardness MAE test: '+str(Hardness_MAE_test))
print ('************************************************')

plt.figure(figsize=(14,6))

fit = np.linspace(min(y1_train),max(y1_train),100)

plt.subplot(1, 2, 1)
plt.plot(y1_train, mupred1_train , 'o', color='blue', label='train', markersize=4)
plt.plot(y1_test, mupred1_test , 'o', color='red', label='test', markersize=4)
plt.plot(fit, fit, 'x', label='', color='black',markersize=2)
plt.xlabel('Experimental hardness', fontsize=12)
plt.ylabel('Mean Predicted hardness', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.errorbar(y1_train, mupred1_train, yerr=errbarpred1_train.flatten(), fmt='o', color='blue',markersize=4, label='train')
plt.errorbar(y1_test, mupred1_test, yerr=errbarpred1_test.flatten(), fmt='o', color='red',markersize=4, label='test')
plt.plot(fit, fit, 'x', label='', color='black', markersize=2)
plt.xlabel('Experimental hardness', fontsize=12)
plt.ylabel('Predicted hardness', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(os.path.join(resultdir,str(seed)+'-'+'Hardness.pdf'),dpi=300)
plt.show()
plt.close()

print('MFGP Predictions plots: Strength')
Strength_MRelAE_train = mean_relative_absolute_error( y2_train, mupred2_train )
Strength_MRelAE_test = mean_relative_absolute_error( y2_test, mupred2_test )

Strength_MAE_train = mean_absolute_error( y2_train,  mupred2_train )
Strength_MAE_test = mean_absolute_error( y2_test,  mupred2_test )

print ('************************************************')
print ('Strength MRel.AE train: '+str(Strength_MRelAE_train))
print ('Strength MRel.AE test: '+str(Strength_MRelAE_test))

print ('Strength MAE train: '+str(Strength_MAE_train))
print ('Strength MAE test: '+str(Strength_MAE_test))
print ('************************************************')

plt.figure(figsize=(14,6))

fit = np.linspace(min(y2_train),max(y2_train),100)

plt.subplot(1, 2, 1)
plt.plot(y2_train, mupred2_train , 'o', color='blue', label='train', markersize=4)
plt.plot(y2_test, mupred2_test , 'o', color='red', label='test', markersize=4)
plt.plot(fit, fit, 'x', label='', color='black',markersize=2)
plt.xlabel('Experimental strength', fontsize=12)
plt.ylabel('Mean Predicted strength', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.errorbar(y2_train, mupred2_train, yerr=errbarpred2_train.flatten(), fmt='o', color='blue',markersize=4, label='train')
plt.errorbar(y2_test, mupred2_test, yerr=errbarpred2_test.flatten(), fmt='o', color='red',markersize=4, label='test')
plt.plot(fit, fit, 'x', label='', color='black', markersize=2)
plt.xlabel('Experimental strength', fontsize=12)
plt.ylabel('Predicted strength', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(os.path.join(resultdir,str(seed)+'-'+'Strength.pdf'),dpi=300)
plt.show()
plt.close()


# ## Diagnostic checks

# In[15]:


postsamples1_train = postsam1_train[:,0,:].T
postsamples1_test = postsam1_test[:,0,:].T
postsamples1_train.shape, postsamples1_test.shape


# In[16]:


postsamples2_train = postsam2_train[:,0,:].T
postsamples2_test = postsam2_test[:,0,:].T
postsamples2_train.shape, postsamples2_test.shape


# In[17]:


sigma1 = np.sqrt(m1.Gaussian_noise.variance.values[0])
sigma2 = np.sqrt(m2.Gaussian_noise.variance.values[0])


# In[18]:


# predictive posterior (y1*|y)
predpostsamples1_test = np.zeros((postsamples1_test.shape[0], postsamples1_test.shape[1])) 
for i in range(predpostsamples1_test.shape[0]):
    predpostsamples1_test[i,:] = pm.Normal.dist(mu=postsamples1_test[i,:], 
                                                sigma=sigma1).random()
predpostsamples1_test_un = scaler_y1.inverse_transform(predpostsamples1_test)
print (predpostsamples1_test_un.shape)
print (y1_test.shape)

# https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html
from statsmodels.distributions.empirical_distribution import ECDF

ecdf_y1_test = []
for i in range(y1_test.shape[0]):
    ecdf = ECDF(predpostsamples1_test_un[:,i]) # Build ECDF with all samples
    ecdf_y1_test.append(ecdf(y1_test[i])[0]) # CDF evaluated at test points
    
    
fig, ax = plt.subplots(dpi=100)
ax.plot(ecdf_y1_test, 'o', color='red', markersize=3)
ax.plot(np.arange(y1_test.shape[0]), 0.0 * np.ones(y1_test.shape[0]), 'r--')
ax.plot(np.arange(y1_test.shape[0]), 1.0 * np.ones(y1_test.shape[0]), 'r--')
ax.set_xlabel('$i$')
ax.set_ylabel('$ECDF \ at \ Hardness \ test$');


import scipy.stats as st
fig, ax = plt.subplots(dpi=100)
st.probplot(ecdf_y1_test, dist=st.uniform, plot=ax)
ax.set_title('Probability plot - Hardness test')
ax.get_lines()[1].set_color('black')
ax.get_lines()[0].set_color('red')
ax.get_lines()[0].set_markersize(3.0);
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Ordered values')
plt.savefig(os.path.join(resultdir,str(seed)+'-'+'Probability_plot_Hardness_test.pdf'),dpi=300)

# https://www.geeksforgeeks.org/ml-kolmogorov-smirnov-test/
res = st.kstest(ecdf_y1_test,"uniform")    
print(res)

zs = ecdf_y1_test
fig, ax = plt.subplots(dpi=100)
ks = np.linspace(-0.5, 1.5, 100)
ax.plot(ks, st.uniform.pdf(ks))
ax.hist(zs, density=True, alpha=0.5, bins = 30)
ax.set_xlabel('$z_i$')


# In[19]:


# predictive posterior (y2*|y)
predpostsamples2_test = np.zeros((postsamples2_test.shape[0], postsamples2_test.shape[1])) 
for i in range(predpostsamples2_test.shape[0]):
    predpostsamples2_test[i,:] = pm.Normal.dist(mu=postsamples2_test[i,:], 
                                                sigma=sigma2).random()
predpostsamples2_test_un = scaler_y2.inverse_transform(predpostsamples2_test)
print (predpostsamples2_test_un.shape)
print (y2_test.shape)

# https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html
from statsmodels.distributions.empirical_distribution import ECDF

ecdf_y2_test = []
for i in range(y2_test.shape[0]):
    ecdf = ECDF(predpostsamples2_test_un[:,i]) # Build ECDF with all samples
    ecdf_y2_test.append(ecdf(y2_test[i])[0]) # CDF evaluated at test points
    
    
fig, ax = plt.subplots(dpi=100)
ax.plot(ecdf_y2_test, 'o', color='red', markersize=3)
ax.plot(np.arange(y2_test.shape[0]), 0.0 * np.ones(y2_test.shape[0]), 'r--')
ax.plot(np.arange(y2_test.shape[0]), 1.0 * np.ones(y2_test.shape[0]), 'r--')
ax.set_xlabel('$i$')
ax.set_ylabel('$ECDF \ at \ Strength \ test$');


import scipy.stats as st
fig, ax = plt.subplots(dpi=100)
st.probplot(ecdf_y2_test, dist=st.uniform, plot=ax)
ax.set_title('Probability plot - Strength test')
ax.get_lines()[1].set_color('black')
ax.get_lines()[0].set_color('red')
ax.get_lines()[0].set_markersize(3.0);
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Ordered values')
plt.savefig(os.path.join(resultdir,str(seed)+'-'+'Probability_plot_Strength_test.pdf'),dpi=300)

# https://www.geeksforgeeks.org/ml-kolmogorov-smirnov-test/
res = st.kstest(ecdf_y2_test,"uniform")    
print(res)

zs = ecdf_y2_test
fig, ax = plt.subplots(dpi=100)
ks = np.linspace(-0.5, 1.5, 100)
ax.plot(ks, st.uniform.pdf(ks))
ax.hist(zs, density=True, alpha=0.5, bins = 30)
ax.set_xlabel('$z_i$')


# In[20]:


print ('Required outputs:')
print ('Seed used to split train and test data: '+str(seed_subset_selection)) 
print ('Strength MRel.AE test: '+str(Strength_MRelAE_test))
print ('Strength MAE test: '+str(Strength_MAE_test))
print ('Strength KS test: '+str(res.statistic))


# In[21]:


# y2_test
sys.stdout = orig_stdout
q.close() 


# In[ ]:
# to run 
# python ModelB-Htest_Unavailable.py -seed=23




