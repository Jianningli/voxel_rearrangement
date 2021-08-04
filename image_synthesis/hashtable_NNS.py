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




class Dictlist(dict):
	'''
	Make a dictionary with duplicate keys in Python
	https://stackoverflow.com/questions/10664856/make-a-dictionary-with-duplicate-keys-in-python
	'''

	def __setitem__(self, key, value):
	    try:
	        self[key]
	    except KeyError:
	        super(Dictlist, self).__setitem__(key, [])
	    self[key].append(value)





class shapeSynthesis(object):

	'''
	Volumetric shape synthesis.

	'''

	def __init__(self):
		self.block_size=3



	def createPyramiddown(self,data):
		pyr=[]
		for i in range(2):
			data=zoom(data,(1/2,1/2,1/2),order=2, mode='constant')
			pyr.append(data)
		return pyr


		
	def createPyramidup(self,data):
		pyr=[]
		for i in range(2):
			data=zoom(data,(2,2,2),order=2, mode='constant')
			data=np.round(data)
			pyr.append(data)
		return pyr



# add zero-padding for patch extraction
# by default, patch size is 5

	def createSparseblock(self,b):

		margin=self.block_size-1
		step=int(margin/2)
		sparse_hashtable=Dictlist()

		temp_0=np.zeros(shape=(b.shape[0]+margin,b.shape[1]+margin,b.shape[2]+margin))
		temp_0[int(margin/2):int(margin/2)+b.shape[0],int(margin/2):int(margin/2)+b.shape[1],int(margin/2):int(margin/2)+b.shape[2]]=b

		for x in range(step,temp_0.shape[0]-step):
			for y in range(step,temp_0.shape[1]-step):
				for z in range(step,temp_0.shape[2]-step):
					temp=temp_0[x-step:x+step+1,y-step:y+step+1,z-step:z+step+1]
					temp=np.reshape(temp,(self.block_size**3,))
					if np.sum(temp)>0:
						#sparse_hashtable[BitArray(temp).uint]=[x,y,z]
						sparse_hashtable['0b'+BitArray(temp).bin]=[x,y,z]
		return sparse_hashtable





	def hamming_distance(self,a,b):
		return (BitArray(a)^BitArray(b)).count(1)


	'''
	multiple keys with the same value
	https://stackoverflow.com/questions/2974022/is-it-possible-to-assign-the-same-value-to-multiple-keys-in-a-dict-object-at-onc
	'''


	def NN_compute(self,dict_):

		new_dict = {}


		for key,index in dict_.items():
			key=BitArray(key)

			temp=key
			'''
			hamming distance 1
			'''

			for i in range(self.block_size**3):
				key=temp
				key[i] = not key[i]

				new_dict.update(dict.fromkeys(['0b'+key.bin],index))


			'''
			hamming distance 2
			'''
			key=temp
			for i in range(self.block_size**3):
				for j in range(i+1,self.block_size**3):
					key=temp
					key[i] = not key[i]
					key[j] = not key[j]
					new_dict.update(dict.fromkeys(['0b'+key.bin],index))


			'''
			hamming distance 3

			key=temp
			for i in range(self.block_size**3):
				for j in range(i+1,self.block_size**3):
					for k in range(j+1,self.block_size**3):
						key=temp
						key[i] = not key[i]
						key[j] = not key[j]
						key[k] = not key[k]
						new_dict.update(dict.fromkeys(['0b'+key.bin],index))


	
			hamming distance 4
			key=temp
			for i in range(self.block_size**3):
				for j in range(i+1,self.block_size**3):
					for k in range(j+1,self.block_size**3):
						for l in range(k+1,self.block_size**3):
							key=temp
							key[i] = not key[i]
							key[j] = not key[j]
							key[k] = not key[k]
							key[l] = not key[l]
							new_dict.update(dict.fromkeys(['0b'+key.bin],index))
			'''
							
		print('length of the nearest neighbors',len(new_dict.keys()))

		return new_dict






	def Synthesis(self,overallList):
		reconstruction=np.zeros(shape=(256,256,32))

		rec_ht=overallList[0]
		db_ht_search=overallList[1]
		db_ht_copy=overallList[2]
		print('number of queries:',len(rec_ht))

		n_match=0
		n_1_match=0
		t=0

		print('pre-computing nearest neighbors...')
		nn_dict=self.NN_compute(db_ht_search)
		print('computing done...')


		for key,index in rec_ht.items():
			t+=1
			print(t)
			if key in db_ht_search.keys():
				n_match+=1
				ii=db_ht_search.get(key)
			
			elif key in nn_dict.keys():
				n_1_match+=1
				ii=nn_dict.get(key)

			#else:

				'''
				approximate NN search


				for k in db_ht_search.keys():
					hmd=self.hamming_distance(k,key)
					if hmd <= 5:
						ii=db_ht_search.get(k)
				'''
			'''
			if len(ii[0][0])>1:
				all_element_with_the_key=[db_ht_copy[i[0]-2,i[1]-2,i[2]-2] for i in ii[0][0]]
				sum_of_element=np.sum(all_element_with_the_key)
				print('sum_of_element',sum_of_element)
				print('len(ii)',len(ii[0][0]))

				if sum_of_element>=len(ii[0][0]):
					holder=1
				else:
					holder=0

			else:
				x=ii[0][0][0][0]-2
				y=ii[0][0][0][1]-2
				z=ii[0][0][0][2]-2
				holder=db_ht_copy[x,y,z]

			'''

			x=ii[0][0][0][0]-1
			y=ii[0][0][0][1]-1
			z=ii[0][0][0][2]-1
			holder=db_ht_copy[x,y,z]


			for idx in index[0][0]:
				reconstruction[idx[0]-1,idx[1]-1,idx[2]-1]=holder

		print('matching rate:',n_match/len(rec_ht))
		print('neighbor 3 matching rate:',n_1_match/len(rec_ht))
		print('length of the precomputed nearest neighbor:',len(nn_dict.keys()))

		return reconstruction





	@staticmethod
	def loadImages(retrivedImg,reconstructedImg):
		print('loading images...')
		r1,h1=nrrd.read(retrivedImg)
		#r1,h1=nrrd.read('000.nrrd')
		r1_1=r1[:,:,0:64]
		r1_2=r1[:,:,64:64*2]
		r1_3=r1[:,:,64*2:64*3]
		r1_4=r1[:,:,64*3:64*4]

		#(128,128,64)
		reconstructed,h=nrrd.read(reconstructedImg)			
		#reconstructed,h=nrrd.read('skull0.nrrd')
		reconstructed_1=reconstructed[:,:,0:16]
		reconstructed_2=reconstructed[:,:,16:16*2]
		reconstructed_3=reconstructed[:,:,16*2:16*3]
		reconstructed_4=reconstructed[:,:,16*3:16*4]
		return [r1_1,r1_2,r1_3,r1_4],[reconstructed_1,reconstructed_2,reconstructed_3,reconstructed_4],h



	#@staticmethod
	def createPyramid(self,retList,recList):
		print('creating pyramid...')
		pool_pyr = Pool(os.cpu_count())
		pyr_db=pool_pyr.map(self.createPyramiddown, [retList[0], retList[1], retList[2],retList[3]])
		pyr_rec=pool_pyr.map(self.createPyramidup, [recList[0], recList[1], recList[2],recList[3]])
		return pyr_db, pyr_rec



	#@staticmethod
	def createSparseBlock(self,pyr_db, pyr_rec):
		print('creating sparse blocks...')
		pool_sparse = Pool(os.cpu_count())
		sparse_db=pool_sparse.map(self.createSparseblock, [pyr_db[0][0],pyr_db[1][0],pyr_db[2][0],pyr_db[3][0]])
		sparse_rec=pool_sparse.map(self.createSparseblock, [pyr_rec[0][0],pyr_rec[1][0],pyr_rec[2][0],pyr_rec[3][0]])
		return sparse_db, sparse_rec


	#@staticmethod
	def synthesis256(self,sparse_db,sparse_rec,pyr_db):		
		pool_synthesis = Pool(250)
		list1=[sparse_rec[0],sparse_db[0],pyr_db[0][0]]
		list2=[sparse_rec[1],sparse_db[1],pyr_db[1][0]]
		list3=[sparse_rec[2],sparse_db[2],pyr_db[2][0]]
		list4=[sparse_rec[3],sparse_db[3],pyr_db[3][0]]
		rec_syn256=pool_synthesis.map(self.Synthesis, [list1,list2,list3,list4])
		return rec_syn256


	#@staticmethod
	def reconstruct256(self,rec_syn256,h):
		container=np.zeros(shape=(256,256,128))
		container[:,:,0:32]=rec_syn256[0]
		container[:,:,32:32*2]=rec_syn256[1]
		container[:,:,32*2:32*3]=rec_syn256[2]
		container[:,:,32*3:32*4]=rec_syn256[3]
		nrrd.write('0_32_synthesized.nrrd',container,h)



if __name__ == "__main__":
	#original: (512, 512, 256)
	#level 0:  (256, 256, 128)
	#level 1:  (128, 128, 64)
	f1='D:/global_local_feature/selecteddata/gt/032.nrrd'
	f2='D:/global_local_feature/selecteddata/inial_pred/skull00.nrrd'
	ss=shapeSynthesis()
	retList,recList,h=ss.loadImages(f1,f2)
	pyr_db, pyr_rec=ss.createPyramid(retList,recList)
	sparse_db, sparse_rec=ss.createSparseBlock(pyr_db,pyr_rec)
	rec_syn256=ss.synthesis256(sparse_db,sparse_rec,pyr_db)
	ss.reconstruct256(rec_syn256,h)




