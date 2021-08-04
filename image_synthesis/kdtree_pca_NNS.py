from __future__ import division
import nrrd
import numpy as np
import scipy
import scipy.ndimage
import random
from scipy.ndimage import zoom
import time
from bitarray import bitarray
from bitstring import BitArray, BitStream
from multiprocessing import Pool
import os
import sklearn
import sklearn.neighbors
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter


# kd tree+pca+multiprocessing

def create_pyramid_down(data):
	pyr=[]
	for i in range(2):
		data=zoom(data,(1/2,1/2,1/2),order=2, mode='constant')
		pyr.append(data)
	return pyr


	
def create_pyramid_up(data):
	pyr=[]
	for i in range(2):
		data=zoom(data,(2,2,2),order=2, mode='constant')
		data=np.round(data)
		pyr.append(data)
	return pyr




def create_sparse_block(b):
	b1=b[0]
	b2=b[1]
	block_size1=5
	block_size2=3
	margin=4
	temp_0=np.zeros(shape=(b1.shape[0]+margin,b1.shape[1]+margin,b1.shape[2]+margin))
	temp_0[int(margin/2):int(margin/2)+b1.shape[0],int(margin/2):int(margin/2)+b1.shape[1],int(margin/2):int(margin/2)+b1.shape[2]]=b1

	margin1=2
	temp_1=np.zeros(shape=(b2.shape[0]+margin1,b2.shape[1]+margin1,b2.shape[2]+margin1))
	temp_1[int(margin1/2):int(margin1/2)+b2.shape[0],int(margin1/2):int(margin1/2)+b2.shape[1],int(margin1/2):int(margin1/2)+b2.shape[2]]=b2


	step=int((block_size1-1)/2)
	stepp=int((block_size2-1)/2)
	sparse_block=[]
	voxel_corr=[]
	for x in range(step,temp_0.shape[0]-step):
		for y in range(step,temp_0.shape[1]-step):
			for z in range(step,temp_0.shape[2]-step):
				temp=temp_0[x-step:x+step+1,y-step:y+step+1,z-step:z+step+1]
				#if np.sum(temp)>0 and np.sum(temp)<block_size*block_size*block_size:
				if np.sum(temp)>0:
					voxel_corr.append([x,y,z])
					xx=x // 2
					yy=y // 2
					zz=z // 2
					tempp=temp_1[xx-stepp:xx+stepp+1,yy-stepp:yy+stepp+1,zz-stepp:zz+stepp+1]
					tempp_=np.reshape(tempp,(block_size2*block_size2*block_size2))
					temp_=np.reshape(temp,(block_size1*block_size1*block_size1))
					temppp=np.concatenate((temp_,tempp_))
					sparse_block.append(temppp)
	finalret=[sparse_block,voxel_corr]
	return finalret





def synthesis_0(overallList):
	temp=np.zeros(shape=(256,256,32))
	blk_rec0=overallList[0]
	idx_rec0=overallList[1]
	blk_db0=overallList[2]
	idx_db0=overallList[3]
	pyr_db=overallList[4]

	print('number of queries:',len(idx_rec0))
	print('create KD tree....................')
	blk_db0=np.array(blk_db0)
	blk_rec0=np.array(blk_rec0)

	pca = PCA(n_components=20)
	blk_db0_pca = pca.fit_transform(blk_db0)
	blk_rec0_pca = pca.transform(blk_rec0)

	tree=KDTree(blk_db0_pca)
	print('search..............')
	for i in range(len(idx_rec0)):
		print(i)
		dist,ind = tree.query(np.reshape(blk_rec0_pca[i],(1,20)),k=1)
		#dist,ind = tree.query(np.reshape(blk_rec0[i],(1,5*5*5+3*3*3)),k=1)
		print('hamming distance:',np.sum(abs(blk_rec0[i]-blk_db0[ind])))
		ind=int(ind)
		xx=idx_rec0[i][0]-2
		yy=idx_rec0[i][1]-2
		zz=idx_rec0[i][2]-2

		x=idx_db0[ind][0]-2
		y=idx_db0[ind][1]-2
		z=idx_db0[ind][2]-2
		temp[xx,yy,zz]=pyr_db[x,y,z]
	return temp



if __name__ == "__main__":
	#original: (512, 512, 256)
	#level 0:  (256, 256, 128)
	#level 1:  (128, 128, 64)

	f1='../template_skull.nrrd'
	f2='../coarse_skull.nrrd'

	print('# ***************load Retrieved images*********')
	#(512,512,256)
	r1,h1=nrrd.read(f1)
	r1_1=r1[:,:,0:64]
	r1_2=r1[:,:,64:64*2]
	r1_3=r1[:,:,64*2:64*3]
	r1_4=r1[:,:,64*3:64*4]


	print('# ***************load reconstructed image********')
	#(128,128,64)
	reconstructed,h=nrrd.read(f2)
	reconstructed_1=reconstructed[:,:,0:16]
	reconstructed_2=reconstructed[:,:,16:16*2]
	reconstructed_3=reconstructed[:,:,16*2:16*3]
	reconstructed_4=reconstructed[:,:,16*3:16*4]


	print('#****************create pyramid*************')
	pool_pyr = Pool(os.cpu_count())
	pyr_db=pool_pyr.map(create_pyramid_down, [r1_1, r1_2, r1_3,r1_4])
	pyr_rec=pool_pyr.map(create_pyramid_up, [reconstructed_1, reconstructed_2, reconstructed_3,reconstructed_4])




	print('#****************create sparse blocks*************')
	pool_sparse = Pool(os.cpu_count())

	list1db=[pyr_db[0][0],pyr_db[0][1]]
	list2db=[pyr_db[1][0],pyr_db[1][1]]
	list3db=[pyr_db[2][0],pyr_db[2][1]]
	list4db=[pyr_db[3][0],pyr_db[3][1]]


	list1rec=[pyr_rec[0][0],reconstructed_1]
	list2rec=[pyr_rec[1][0],reconstructed_2]
	list3rec=[pyr_rec[2][0],reconstructed_3]
	list4rec=[pyr_rec[3][0],reconstructed_4]


	sparse_db=pool_sparse.map(create_sparse_block, [list1db,list2db,list3db,list4db])
	sparse_rec=pool_sparse.map(create_sparse_block, [list1rec,list2rec,list3rec,list4rec])


	print('#****************synthesis*************')
	pool_synthesis = Pool(250)
	list1=[sparse_rec[0][0],sparse_rec[0][1],sparse_db[0][0],sparse_db[0][1],pyr_db[0][0]]
	list2=[sparse_rec[1][0],sparse_rec[1][1],sparse_db[1][0],sparse_db[1][1],pyr_db[1][0]]
	list3=[sparse_rec[2][0],sparse_rec[2][1],sparse_db[2][0],sparse_db[2][1],pyr_db[2][0]]
	list4=[sparse_rec[3][0],sparse_rec[3][1],sparse_db[3][0],sparse_db[3][1],pyr_db[3][0]]
	rec_syn256=pool_synthesis.map(synthesis_0, [list1,list2,list3,list4])

	print('#****************reconstruct*************')
	container=np.zeros(shape=(256,256,128))
	container[:,:,0:32]=rec_syn256[0]
	container[:,:,32:32*2]=rec_syn256[1]
	container[:,:,32*2:32*3]=rec_syn256[2]
	container[:,:,32*3:32*4]=rec_syn256[3]
	nrrd.write('reconstructed.nrrd',container,h)


