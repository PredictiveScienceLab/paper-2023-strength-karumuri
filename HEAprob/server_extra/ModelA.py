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


resultdir = os.path.join(os.getcwd(), 'ModelA_results' ,str(seed)+'-seed') 

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


raw_data = pd.read_csv(os.path.join(path,'Strength_Hardness_with_Descriptors-Sh.csv'), index_col=0)
# display(raw_data)


# Extracting all relevant features:

# In[6]:


# The unscaled input and output features
RoM = ['Density', 'Modulus', 'Tm', 'Misfit', 'Rbar', 'delta', 'shear', 'bulk', 'VEC', 'SMIX']
MX = ['ModulusMX', 'DensityMX', 'TmMX', 'R_MX', 'shearMX', 'bulkMX', 'VECMX']
Phase = ['Phase_BCC', 'Phase_BCC+BCC', 'Phase_BCC+Sec.', 'Phase_FCC', 'Phase_FCC+BCC', 'Phase_FCC+Sec.', 'Phase_No Data', 'Phase_Other']
X = RoM + MX + Phase
# Curtin = 'Delta_SS'
H = 'PROPERTY: HV'
S = 'PROPERTY: YS (MPa)'


# ## ModelA

# In this Case, we are going to build a Strength model using GP

# using {X=[X(above),H], Y=S}, mean prediction = S_pred 

# In[7]:


print ('Splitting')
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(raw_data, test_size=0.2, random_state=seed_subset_selection)
print (data_train.shape, data_test.shape)
# print (data_train[:2])
# print (data_test[:2])


# In[8]:


print ('Pre-processing the data for GP2 building')
print ('Step1: Seperating')
X_train, y1_train, y2_train = data_train[X].values, data_train[H].values[:, None], data_train[S].values[:, None] 
X_test, y1_test, y2_test = data_test[X].values, data_test[H].values[:, None], data_test[S].values[:, None] 

print('Step2: Scaling [X, H], Scaling S')
# scale train data and use those means and dev for scaling test data to avoid info leakage
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(X_train)
X_train_s, X_test_s = scaler_X.transform(X_train), scaler_X.transform(X_test)
scaler_y1 = StandardScaler()
scaler_y1.fit(y1_train)
y1_train_s, y1_test_s = scaler_y1.transform(y1_train), scaler_y1.transform(y1_test)
scaler_y2 = StandardScaler()
scaler_y2.fit(y2_train)
y2_train_s, y2_test_s = scaler_y2.transform(y2_train), scaler_y2.transform(y2_test)


# In[9]:


print ('Training and testing dataset sizes')
print (np.shape(X_train_s), np.shape(y1_train_s), np.shape(y2_train_s))
print (np.shape(X_test_s), np.shape(y1_test_s), np.shape(y2_test_s))


# In[10]:


print ('Building GP2 between [X_s,H_s] and S_s')
k2 = GPy.kern.RBF(np.hstack((X_train_s,y1_train_s)).shape[1], ARD=True)
m2 = GPy.models.GPRegression(np.hstack((X_train_s,y1_train_s)), y2_train_s, k2)
m2.optimize_restarts()
print(m2)
print(m2[".*Gaussian_noise"])
print(m2[".*rbf"])


# In[11]:


print('GP Predictions') # posterior (f2|y) and (f2*|y)
postsam2_train = m2.posterior_samples_f(np.hstack((X_train_s,y1_train_s)), size=1000) # posterior_samples_f gives epistemic uncertainity only
postsam2_train_un = scaler_y2.inverse_transform(postsam2_train) # un-normalized values
print (postsam2_train_un.shape)

postsam2_test = m2.posterior_samples_f(np.hstack((X_test_s,y1_test_s)), size=1000) # posterior_samples_f gives epistemic uncertainity only
postsam2_test_un = scaler_y2.inverse_transform(postsam2_test) # un-normalized values 
print (postsam2_test_un.shape)

mupred2_train, varpred2_train = np.mean(postsam2_train_un, axis=2), np.var(postsam2_train_un, axis=2)
mupred2_test, varpred2_test = np.mean(postsam2_test_un, axis=2), np.var(postsam2_test_un, axis=2)

errbarpred2_train = 1.96*np.sqrt(varpred2_train)
errbarpred2_test = 1.96*np.sqrt(varpred2_test)


# In[12]:


from sklearn.metrics import mean_absolute_error
def mean_relative_absolute_error(y_true, y_pred): 
    return np.mean( np.abs((y_pred-y_true)/y_true) )

print('GP Predictions plots')
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

fit = np.linspace(min(raw_data[S]),max(raw_data[S]),100)

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

# If the model is correct i.e., if the statistics predicted by model is correct, then ECDF evaluated at the test data should follow a uniform distribution.

# In[13]:


postsamples2_train = postsam2_train[:,0,:].T
postsamples2_test = postsam2_test[:,0,:].T
print(postsamples2_train.shape, postsamples2_test.shape)


# In[14]:


sigma2 = np.sqrt(m2.Gaussian_noise.variance.values[0])
print(sigma2)


# In[15]:


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


# In[16]:


print ('Required outputs:')
print ('Seed used to split train and test data: '+str(seed_subset_selection)) 
print ('Strength MRel.AE test: '+str(Strength_MRelAE_test))
print ('Strength MAE test: '+str(Strength_MAE_test))
print ('Strength KS test: '+str(res.statistic))


# In[17]:


# y2_test
sys.stdout = orig_stdout
q.close() 


# In[ ]:
# to run 
# python ModelA.py -seed=23




