### Learning to Rearrange Voxels in BinarySegmentation Masks for Smooth Manifold Triangulation



Dataset: [AutoImplant 2021 Challenge Task 3 Dataset](https://autoimplant2021.grand-challenge.org/Dataset/)



**denoise.py** : denoise the implant obtained via subtraction based on CCA and morphological openning. Kernel size needs to be adjusted for each case (larger kernel removes more voxels)

```python
def morph(data):
    kernel = np.ones((9,9),np.uint8)
    mo=cv2.morphologyEx(data.astype('uint8'), cv2.MORPH_OPEN, kernel)
    return mo
```



```python
zl=zz//64
patch_idx=0
t1=time.time()
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
