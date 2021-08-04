import numpy as np
import random
import nrrd


# Data loader for patch-wise shape completion and patch-wise voxel rearrangement



def make_random_patch(data,label):
    x,y,z=data.shape
    idx=random.randrange(0,512-128,1)
    idy=random.randrange(0,512-128,1)
    idz=random.randrange(0,z-64,1)
    temp_data=np.expand_dims(np.expand_dims(data[idx:idx+128,idy:idy+128,idz:idz+64],axis=0),axis=4)
    temp_label=np.expand_dims(np.expand_dims(label[idx:idx+128,idy:idy+128,idz:idz+64],axis=0),axis=4)

    return temp_data,temp_label




# for training model
def load_random_patch_pair(list1,list2):
   # generate a batch of paired patches for training
    idx=random.randrange(0,100,1)
    data,h=nrrd.read(list1[idx])
    print('data',list1[idx])
    label,h=nrrd.read(list2[idx])
    print('label',list2[idx])

    defected_patch,label_patch=make_random_patch(data,label)
    return defected_patch,label_patch


#**************************** for testing ********************#

def make_random_test_patch(data):
    data_list=[]
    a,b,c=data.shape
    zz=(c//64)
    for x in range(4):
        for y in range(4):
            for z in range(zz):            
                temp_data=np.expand_dims(np.expand_dims(data[x*128:(x+1)*128,y*128:(y+1)*128,z*64:(z+1)*64],axis=0),axis=4)
                data_list.append(temp_data)

    for x in range(4):
        for y in range(4):
            temp_data=np.expand_dims(np.expand_dims(data[x*128:(x+1)*128,y*128:(y+1)*128,c-64:c],axis=0),axis=4)            
            data_list.append(temp_data)
    return np.array(data_list)




def load_random_test_pair(list1,idx):
   # generate a batch of paired patches for evaluation
    data,h1=nrrd.read(list1[idx])
    print('data',list1[idx])


    #data=pre_processing(data)
    #label=pre_processing(label)
    #implant=pre_processing(implant)    


    test_patch=make_random_test_patch(data)
    return test_patch, h1, data.shape[2]




