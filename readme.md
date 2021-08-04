### Learning to Rearrange Voxels in BinarySegmentation Masks for Smooth Manifold Triangulation



Dataset: [AutoImplant 2021 Challenge Task 3 Dataset](https://autoimplant2021.grand-challenge.org/Dataset/)



**denoise.py** : denoise the implant obtained via subtraction based on CCA and morphological openning. Kernel size needs to be adjusted for each case (larger kernel removes more voxels)

```python
def morph(data):
    kernel = np.ones((9,9),np.uint8)
    mo=cv2.morphologyEx(data.astype('uint8'), cv2.MORPH_OPEN, kernel)
    return mo
```
