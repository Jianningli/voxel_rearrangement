### Hierarchical Image synthesis using Hash table and KD-tree-based Nearest Neighbor Search (NNS)






Hash table equals to the _dictionary_ in Python. The base _dict_ class used to create the hash tables for the template and coarse pyramid is as follows:
```python
class Dictlist(dict):
	
	#Make a dictionary with duplicate keys in Python
	#https://stackoverflow.com/questions/10664856/make-a-dictionary-with-duplicate-keys-in-python
	

	def __setitem__(self, key, value):
	    try:
	        self[key]
	    except KeyError:
	        super(Dictlist, self).__setitem__(key, [])
	    self[key].append(value)
```



Hamming distance between two bit strings:  ^ stands for Binary XOR. function _.count(1)_ counts the number of '1s' in the bit string.
```python
(bitStr1^bitStr2).count(1)
```


Build a hash table using the coordinates and bit strings. function _BitArray(binaryArray).bin_ converts a binary array to a bit string.
```python
sparse_hashtable['0b'+BitArray(temp).bin]=[x,y,z]
```


Function _def NN_compute(self,dict_)_ pre-computes the bit string neighbors (not to be confused with the voxel neighbors). The larger the neighber size, the longer it takes for the pre-computation.
