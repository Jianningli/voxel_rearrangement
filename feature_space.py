import tensorflow as tf
from data_loader import *
import nrrd
import os
from scipy.special import softmax
import numpy as np
from glob import glob
import time
import scipy.ndimage

class ImportGraph(object):
	def __init__(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.per_process_gpu_memory_fraction = 0.90 
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph,config=self.config)

		#self.input_I = tf.placeholder(dtype=tf.float32, shape=[1,128,128,32,1], name='inputI')
		with self.graph.as_default():
			saver = tf.train.import_meta_graph('auto_encoder.meta')
			saver.restore(self.sess, 'auto_encoder')
			
			# print all the tensors in the graph
			#for tensor in tf.get_default_graph().get_operations():
			#	print (tensor)
            
            #print all the operation names in the graph
			#for op in tf.get_default_graph().get_operations():
			#	print(op.name)


			#self.Conv1_1 = tf.get_default_graph().get_operation_by_name('Conv1_1/relu').outputs[0]
			#print(self.Conv1_1.shape)
			#self.conv1_relu = tf.get_default_graph().get_operation_by_name('conv1_relu').outputs[0]
			#print(self.conv1_relu.shape)
			#self.conv2_relu = tf.get_default_graph().get_operation_by_name('conv2_relu').outputs[0]
			#print(self.conv2_relu.shape)
			#self.conv3_2_relu = tf.get_default_graph().get_operation_by_name('conv3_2_relu').outputs[0]
			#print(self.conv3_2_relu.shape)
			#self.conv4_2_relu = tf.get_default_graph().get_operation_by_name('conv4_2_relu').outputs[0]
			#print(self.conv4_2_relu.shape)
			#self.deconv1_1 = tf.get_default_graph().get_operation_by_name('deconv1_1/relu').outputs[0]
			#print(self.deconv1_1.shape)
			#self.deconv2_1 = tf.get_default_graph().get_operation_by_name('deconv2_1/relu').outputs[0]
			#print(self.deconv2_1.shape)
			#self.deconv3_1 = tf.get_default_graph().get_operation_by_name('deconv3_1/relu').outputs[0]
			#print(self.deconv3_1.shape)
			#self.deconv4_1 = tf.get_default_graph().get_operation_by_name('deconv4_1/relu').outputs[0]
			#print(self.deconv4_1.shape)

			self.pred_prob = tf.get_default_graph().get_operation_by_name('pred_prob/Conv3D').outputs[0]
			print(self.pred_prob.shape)
			self.conv5_1 = tf.get_default_graph().get_operation_by_name('conv5_1/relu').outputs[0]
			print(self.conv5_1.shape)

			#self.pred = tf.get_default_graph().get_operation_by_name('argmax').outputs[0]
			#print(self.pred.shape)


	def run(self, data):
		# tensor=operation:0

		#Conv1_1_out=self.sess.run(self.Conv1_1, feed_dict={'inputI:0':data})
		#print('1')
		#conv1_relu_out=self.sess.run(self.conv1_relu, feed_dict={'inputI:0':data})
		#print('2')
		#conv2_relu_out=self.sess.run(self.conv2_relu, feed_dict={'inputI:0':data})
		#print('3')		
		#conv3_2_relu_out=self.sess.run(self.conv3_2_relu, feed_dict={'inputI:0':data})
		#print('4')
		#conv4_2_relu_out=self.sess.run(self.conv4_2_relu, feed_dict={'inputI:0':data})
		#print('5')

		#print('6')
		#deconv1_1_out=self.sess.run(self.deconv1_1, feed_dict={'inputI:0':data})
		#print('7')
		#deconv2_1_out=self.sess.run(self.deconv2_1, feed_dict={'inputI:0':data})
		#print('8')
		#deconv3_1_out=self.sess.run(self.deconv3_1, feed_dict={'inputI:0':data})
		#print('9')
		#deconv4_1_out=self.sess.run(self.deconv4_1, feed_dict={'inputI:0':data})
		#print('10')
		pred_prob_out=self.sess.run(self.pred_prob, feed_dict={'inputI:0':data})
		#print('11')
		#pred_out=self.sess.run(self.pred, feed_dict={self.input_I:data})

		conv5_1_out=self.sess.run(self.conv5_1, feed_dict={'inputI:0':data})

		return conv5_1_out,pred_prob_out



'''
(1, 128, 128, 32, 128)
(1, 64, 64, 16, 128)
(1, 32, 32, 8, 256)
(1, 16, 16, 4, 256)
(1, 8, 8, 2, 256)
(1, 8, 8, 2, 512)
(1, 16, 16, 4, 512)
(1, 32, 32, 8, 256)
(1, 64, 64, 16, 256)
(1, 128, 128, 32, 128)
(1, 128, 128, 32, 2)
'''


model = ImportGraph()


# feature representation of the database
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
database_feature=[]
database_dir="C:/Users/Jianning/Desktop/CAMedProject/MICCAI2020/newMiccai/database"
pair_list=glob('{}/*.nrrd'.format(database_dir))

'''
pointer=0
for i in range(len(pair_list)):
	data,header = nrrd.read(pair_list[i])
	data=data[:,:,data.shape[2]-128:data.shape[2]]
	data=resizing(data)
	label_hidden,label_out = model.run(np.expand_dims(np.expand_dims(data,axis=0),axis=4))
	#print(label_hidden.shape)
	database_feature.append([pointer,label_hidden[0,:,:,:,:]])
	pointer=pointer+1
	#(1, 8, 8, 2, 512)
	#print(r.shape)





database_feature=np.array(database_feature)

np.save('database_feature.npy',database_feature)

#print(database_feature.shape)
#feature idx
#print(database_feature[0][0])
#feature matrix
#print(database_feature[0][1].shape)

'''

database_feature=np.load('database_feature.npy')

start_time = time.time()
testimg='C:/Users/Jianning/Desktop/CAMedProject/MICCAI2020/newMiccai/test/A0152.nrrd'

test,h=nrrd.read(testimg)
test=test[:,:,test.shape[2]-128:test.shape[2]]
test=resizing(test)
test_hidden,pred_label = model.run(np.expand_dims(np.expand_dims(test,axis=0),axis=4))
### Test whether the graph and weights has been loaded correctly ###
out=softmax(pred_label[0,:,:,:,:])
out=np.argmax(out,axis=3)
#nrrd.write('output.nrrd',out.astype('float32'),header)

out_hidden,out_label = model.run(np.expand_dims(np.expand_dims(out,axis=0),axis=4))

diff=[]
for i in range(len(database_feature)):
	dist=out_hidden[0]-database_feature[i][1]
	dist_=np.sum(np.sqrt(np.sum(np.square(dist), axis=3)))
	diff.append(dist_)


temp=diff
#print('original list',temp)
diff=np.array(diff)
sorted_list=np.sort(diff)
#print('sorted list',sorted_list)
d_idx1=temp.index(sorted_list[0])
d_idx2=temp.index(sorted_list[1])
d_idx3=temp.index(sorted_list[2])
d_idx4=temp.index(sorted_list[3])

print('retrieved database skull',pair_list[d_idx1][-10:-5])
print('retrieved database skull',pair_list[d_idx2][-10:-5])
print('retrieved database skull',pair_list[d_idx3][-10:-5])
#print('retrieved database skull',pair_list[d_idx4][-10:-5])

end_time=time.time()
#gives the execution time in seconds
print('retrieval time:', end_time-start_time)

start_time1 = time.time()

def resizing_back(label):
    a,b,c=label.shape
    resized_data = zoom(label,(512/a,512/b,128/c),order=1, mode='constant')  
    return resized_data

upsampled=resizing_back(out)
#nrrd.write('upsampled.nrrd',upsampled.astype('float32'),header)





ref1,h1=nrrd.read(pair_list[d_idx1])
ref1=ref1[:,:,ref1.shape[2]-128:ref1.shape[2]]
ref2,h2=nrrd.read(pair_list[d_idx2])
ref2=ref2[:,:,ref2.shape[2]-128:ref2.shape[2]]
ref3,h3=nrrd.read(pair_list[d_idx3])
ref3=ref3[:,:,ref3.shape[2]-128:ref3.shape[2]]



def cal_dist(in1,in2):
	dist=in1-in2
	dist_=np.sum(np.sqrt(np.sum(np.square(dist),axis=3)))
	return dist_






#(512,512,128)
reconstructed=np.zeros(shape=(512,512,128))
def cal_dice(pred,input_gt):
	ints_f = np.sum(((input_gt==1)*1)*((pred==1)*1))
	union_f = np.sum(((input_gt==1)*1) + ((pred==1)*1)) + 0.0001
	dice_f=(2.0*ints_f)/union_f

	ints_b = np.sum(((input_gt==0)*1)*((pred==0)*1))
	union_b = np.sum(((input_gt==0)*1) + ((pred==0)*1)) + 0.0001
	dice_b=(2.0*ints_b)/union_b
	return dice_f


def cal_dist1(in1,in2):
	dist=in1-in2
	dist_=np.sqrt(np.sum(np.square(dist)))
	return dist_


patch_size=4

for x in range(int(512*2/patch_size-1)):
	#idx=int((patch_size*x)/2)
	#idx_u=int(idx+patch_size)
	for y in range(int(512*2/patch_size-1)):
		#idy=int((patch_size*y)/2)
		#idy_u=int(idy+patch_size)
		for z in range(int(128*2/patch_size-1)):
			#idz=int((patch_size*z)/2)
			#idz_u=int(idz+patch_size)

			temp_unsampled=upsampled[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size,z*patch_size:(z+1)*patch_size]
			temp_ref1=ref1[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size,z*patch_size:(z+1)*patch_size]
			temp_ref2=ref2[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size,z*patch_size:(z+1)*patch_size]
			temp_ref3=ref3[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size,z*patch_size:(z+1)*patch_size]

			#temp_unsampled=upsampled[idx:idx_u,idy:idy_u,idz:idz_u]
			#temp_ref1=ref1[idx:idx_u,idy:idy_u,idz:idz_u]
			#temp_ref2=ref2[idx:idx_u,idy:idy_u,idz:idz_u]
			#temp_ref3=ref3[idx:idx_u,idy:idy_u,idz:idz_u]			


			ref_patch=[temp_ref1,temp_ref2,temp_ref3]

			dice1=cal_dist1(temp_unsampled,temp_ref1)
			dice2=cal_dist1(temp_unsampled,temp_ref2)
			dice3=cal_dist1(temp_unsampled,temp_ref3)

			dice_list=[dice1,dice2,dice3]
			temp_dist=dice_list
			dist_list_a=np.array(dice_list)
			sorted_dist=np.sort(dist_list_a)
			idxx=temp_dist.index(sorted_dist[0])
			#print(idxx)
			reconstructed[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size,z*patch_size:(z+1)*patch_size]=ref_patch[idxx]
			#reconstructed[idx:idx_u,idy:idy_u,idz:idz_u]=ref_patch[idxx]



end_time1=time.time() 		
print('synthesis time:',end_time1-start_time1)


save_dir='C:/Users/Jianning/Desktop/CAMedProject/MICCAI2020/newMiccai/'
filename=save_dir+testimg[-10:-5]+'.nrrd'
nrrd.write(filename,reconstructed.astype('float32'),h)

filename1=save_dir+testimg[-10:-5]+'_smoothed'+'.nrrd'
smoothed=scipy.ndimage.filters.median_filter(reconstructed,(6,6,6))
nrrd.write(filename1,smoothed.astype('float32'),h)








