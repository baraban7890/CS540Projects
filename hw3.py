from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x,axis = 0)
    return x

def get_covariance(dataset):
    covariance = np.dot(np.transpose(dataset),dataset) / (len(dataset) - 1)
    return covariance

def get_eig(S, m):
    w,v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    v = np.fliplr(v)
    w = np.diag(w)
    w = np.flip(w)
    return w,v

def get_eig_prop(S, prop):
    w,v = eigh(S)
    x,y = eigh(S, subset_by_value = [prop * np.sum(w), np.sum(w)])
    x = np.diag(x)
    x = np.flip(x)
    y = np.fliplr(y)
    return x,y

def project_image(image, U):
    projection = 0
    Ut = np.transpose(U)
    for i in range(len(Ut)):
        weight = np.dot(Ut[i],image)
        projection += np.dot(weight,Ut[i])
        
    return projection

def display_image(orig, proj):
    orig = orig.reshape(32,32)
    orig = np.transpose(orig)
    proj = proj.reshape(32,32)
    proj = np.transpose(proj)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Original')
    ax[1].set_title('Projection')
    plt1 = ax[0].imshow(orig, aspect = 'equal')
    plt2 = ax[1].imshow(proj, aspect = 'equal')
    plt.colorbar(plt1, ax = ax[0])
    plt.colorbar(plt2, ax = ax[1])
    plt.show()
    pass