# -*- coding: utf-8 -*-
# SUBSTITUTE np.nan VALUES IN THIS FILE
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os
import sys
#%%
folderpath = './moles/'
files = os.listdir(folderpath)
#%%
plotfig = True
np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
plt.close('all')# close previously opened pictures
ifile = files.index('low_risk_4.jpg')
filein=files[ifile] # file to be analyzed (low_risk, medium_risk or melanoma)
print(filein)
im_or = mpimg.imread(folderpath+filein)
gray_image = np.mean(im_or, axis=2).astype(np.uint8)# convert to grayscale
N1,N2,N3=im_or.shape # note: N3 is 3, the number of elementary colors, i.e. red, green ,blue
N1,N2=gray_image.shape
# im_or is Ndarray N1 x N2 x 3 unint8 (integers from 0 to 255)
# gray_image is an Ndaaray N1 x N2 unint8 (integers from 0 to 255)
# plot the images, to check them:
plt.figure()
plt.imshow(im_or,interpolation=None)
plt.title('original image')
plt.figure()
plt.imshow(gray_image,cmap='gray', vmin=0, vmax=255,interpolation=None)
plt.title('gray image')
#%% get a simplified image with only Ncluster colors
# number of clusters/quantized colors we want to have in the simpified image:
Ncluster=3
# instantiate the object K-means:
kmeans = KMeans(n_clusters=Ncluster, random_state=0)
# run K-means on the colors of the gray image (i.e. on the uint8 values):
im_1D = gray_image.reshape((N1*N2,1))
kmeans.fit(im_1D)
# get the centroids (i.e. the 3 gray colors). Note that the centroids
# take real values, we must convert these values to uint8
# to properly see the quantized image
Ncluster=len(kmeans.cluster_centers_)# Warning: it is possible that the found clusters is less than required
kmeans_centroids=kmeans.cluster_centers_.astype('uint8')
# copy im_1D into im_1D_quant and get the quantized image
im_1D_quant = im_1D.copy()
for kc in range(Ncluster):
    im_1D_quant[(kmeans.labels_==kc),:]=kmeans_centroids[kc,:]# substitute the centroid value in the pixels that belong to the cluster
im_quant=im_1D_quant.reshape((N1,N2))

if plotfig:
    plt.figure()
    plt.imshow(im_quant,cmap='gray',interpolation=None)
    plt.title('image with quantized colors (after K-Means')
#%% Preliminary steps to find the contour after the clustering
# 1: find the darkest color found by k-means, since the darkest color
# corresponds to the mole:
centroids=kmeans_centroids
i_col=centroids.argmin() # darkest color corresponds to minimum grayscale value
# 2: define the 2D-array im_clust where in position i,j you have the index of
# the cluster pixel i,j belongs to 
im_clust=kmeans.labels_.reshape(N1,N2)
# 3: find the positions i,j where im_clust is equal to i_col (cluster with the darkest color)
# the 2D Ndarray mole_pos stores the coordinates i,j only of the pixels
# in cluster i_col
mole_pos=np.argwhere(im_clust==i_col) # Ndarray with two columns, storing the index [i,j] of the dark pixels 
#%% Find the likely position of the mole using DBSCAN
epsilon = np.nan
if np.isnan(epsilon):
    print("SET AN APPROPRIATE VALUE FOR epsilon!")
    sys.exit()
M = np.nan
if np.isnan(M):
    print("SET AN APPROPRIATE VALUE FOR M!")
    sys.exit()
clusters= DBSCAN(eps=epsilon,min_samples=M,metric='euclidean').fit(mole_pos)  # fit DBSCAN on the positions of the dark pixels (closeby dark pixels will belong to the same cluster)
id_clusters,count_id_clusters = np.unique(clusters.labels_,return_counts=True)# count the number of obtained clusters (i.e. groups of closeby dark pixels)
print('Number of points in each cluster found by DBSCAN: ',count_id_clusters)
print('Indexes of the found clusters: ',id_clusters)
# select the clusters that could potentially correspond to the mole
i_mole = np.nan
if np.isnan(i_mole):
    print("SET AN APPROPRIATE VALUE FOR i_mole (index of the cluster that contains the mole)!")
    sys.exit()


true_mole_pos = mole_pos[clusters.labels_==i_mole] # pixel indexes [i,j] in the cluster that corresponds to the mole
im_only_mole_gray = 0*gray_image-1# white image
x=true_mole_pos[:,0]
y=true_mole_pos[:,1]
im_only_mole_gray[x,y]=gray_image[x,y]
if plotfig:
    plt.figure()
    plt.imshow(im_only_mole_gray,cmap='gray',interpolation=None)
    plt.title('original size image, segmented grayscale mole (after DBSCAN)')
im_mole_pos = np.ones((N1,N2),dtype='uint8')*255# white image
im_mole_pos[x,y]=0 # black where the mole is present
if plotfig:
    plt.figure()
    plt.imshow(im_mole_pos,cmap='gray',interpolation=None)
    plt.title('original size image, mole position')

#%% Find the cropped original image
margin = np.nan # pixels around the mole (used for smoothing the image)
if np.isnan(margin):
    print("SET AN APPROPRIATE VALUE FOR margin!")
    sys.exit()
min_x = np.min(true_mole_pos[:,0])-margin
max_x = np.max(true_mole_pos[:,0])+margin
min_y = np.min(true_mole_pos[:,1])-margin
max_y = np.max(true_mole_pos[:,1])+margin
im_cropped_gray_red = im_only_mole_gray[min_x:max_x,min_y:max_y]
im_cropped_col = im_or[min_x:max_x,min_y:max_y,:]
im_cropped_mole_pos = im_mole_pos[min_x:max_x,min_y:max_y]

if plotfig:
    plt.figure()
    plt.imshow(im_cropped_mole_pos,cmap='gray',interpolation=None)
    plt.title('cropped image, mole position')

#%% Smooth the image
delta = np.nan
if np.isnan(delta):
    print("SET AN APPROPRIATE VALUE FOR delta!")
    sys.exit()
N1,N2=im_cropped_mole_pos.shape
im_cropped_mole_pos_filt = 0*im_cropped_mole_pos+255# white image
for kr in range(delta,N1-delta):
    for kc in range(delta,N2-delta):
        sub = im_cropped_mole_pos[kr-delta:kr+delta,kc-delta:kc+delta]
        im_cropped_mole_pos_filt[kr,kc] = np.median(sub)
if plotfig:     
    plt.figure()      
    plt.imshow(im_cropped_mole_pos_filt,cmap='gray',interpolation=None)  
    plt.title('smoothed cropped image, mole position')      
  

#%% Apply Sobel filters
kern1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])# Sobel filter
kern2= kern1.T
#%% get the border and plot the cropped color image and the border superimposed
border = np.nan
if np.isnan(border):
    print("IMPLEMENT FILTERING WITH SOBEL FILTERS AND FIND THE BORDER")
    sys.exit()
plt.figure()
plt.imshow(im_cropped_col, interpolation='none')
plt.imshow(border,cmap='gray', interpolation='none', alpha=0.2)
