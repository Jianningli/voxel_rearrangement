### Learning to Rearrange Voxels in Binary Segmentation Masks for Smooth Manifold Triangulation [[pdf](https://arxiv.org/pdf/2108.05269.pdf)]



Dataset: [AutoImplant 2021 Challenge Task 3 Dataset](https://autoimplant2021.grand-challenge.org/Dataset/)



**denoise.py** : denoise the implant obtained via subtraction based on CCA and morphological openning. Kernel size needs to be adjusted for each case (larger kernel removes more voxels)

```python
def morph(data):
    kernel = np.ones((9,9),np.uint8)
    mo=cv2.morphologyEx(data.astype('uint8'), cv2.MORPH_OPEN, kernel)
    return mo
```


**loader.py** : Data loader for patch-wise shape completion and patch-wise voxel rearrangement. During testing, using the following function to crop patches (128x128x64) from a skull.

```python
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
 ```

**skull_completion.py** : for skull shape completion on downsampled skulls, patch-wise shape completion and patch-wise voxel rearrangement, the same autoencoder network is used. For testing of the two patch-based methods, the output patches are sequentially combined as follows to form a final complete skull (for each skull, the number of slices, i.e., Z is different).  

```python
reconstructed=np.zeros(shape=(512,512,Z))
zl=num_Z//64
patch_idx=0
for x in range(4):
    for y in range(4):
        for z in range(zl):
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input[patch_idx]})
            reconstructed[x*128:(x+1)*128,y*128:(y+1)*128,z*64:(z+1)*64]=test_output[0,:,:,:]
            patch_idx=patch_idx+1

for x in range(4):
    for y in range(4):
        test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input[patch_idx]})
        reconstructed[x*128:(x+1)*128,y*128:(y+1)*128,zz-64:zz]=test_output[0,:,:,:]
        patch_idx=patch_idx+1
 ```
 
 
 If you find our repository useful or use the codes in your research, please use the following bibtex entry for reference:
 
 ```
 @article{li2021learning,
  title={Learning to Rearrange Voxels in Binary Segmentation Masks for Smooth Manifold Triangulation},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Jin, Yuan and Egger, Jan},
  journal={arXiv preprint arXiv:2108.05269},
  year={2021}
}
 ```
