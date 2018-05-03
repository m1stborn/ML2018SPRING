import numpy as np
from skimage import io
import sys


# imgFile = sys.argv[1]
# path = "./Aberdeen/"

path = sys.argv[1]
targetImg = sys.argv[2] 

imgs = []
x = 0
for i in range(0,415):
	img = io.imread(path+"{}.jpg".format(i))
	img = img.flatten()
	imgs.append(img)
imgs = np.array(imgs).astype('float64')
imgs_mean = np.mean(imgs,axis=0)
imgs -= imgs_mean

targetImg = io.imread(path+targetImg)
targetImg = targetImg.flatten()
targetImg = targetImg.astype('float64')
targetImg -= imgs_mean

# U = np.load('U.npy')
# S = np.load('S.npy')
# V = np.load('V.npy')
U, S, V = np.linalg.svd(imgs.T, full_matrices=False)

eigenface = U.T[:4]

def plotface(i):
	i -= np.min(i)
	i /= np.max(i)
	i = (i*255).astype(np.uint8)
	io.imshow(i.reshape((600,600,3)))
	io.show()
	io.imsave('reconstruction.jpg',i.reshape((600,600,3)))

def eigen_face(eigenface):
	for i in eigenface:
		i -= np.min(i)
		i /= np.max(i)
		i = (i*255).astype(np.uint8)
		io.imshow(i.reshape((600,600,3)))
		io.show()

def reconstruct(U,S,V,imgs,imgs_mean,eigenface):
	weights = np.dot(imgs,U)
	# pics = np.doT(weights,eigenface)
	picked = [5]
	# weights = np.dot(imgs,eigenface.T)

	recons = []
	for i in range(imgs.shape[0]):
		recon = imgs_mean + np.dot(weights[i,:4],U[:,:4].T)
		recons.append(recon)
	recons = np.array(recons)

	for i in recons[picked,:]:
		i -= np.min(i)
		i /= np.max(i)
		i = (i*255).astype(np.uint8)
		io.imshow(i.reshape((600,600,3)))
		io.show()


def recon_slide(U,S,V,imgs,imgs_mean):
	weights = np.dot(imgs,U[:,:4])
	print(weights.shape)
	recon = imgs_mean + np.dot(weights,U[:,:4].T)
	plotface(recon)
	# io.imsave('reconstruction.jpg', recon.reshape((600,600,3)))


# recon_slide(U,S,V,imgs[5],imgs_mean)
recon_slide(U,S,V,targetImg,imgs_mean)
# plotface(targetImg)
# reconstruct(U,S,V,imgs,imgs_mean,eigenface)
# eigen_face(eigen_face)