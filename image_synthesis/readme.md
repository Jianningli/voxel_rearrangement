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



hamming distance between two bit strings:  ^ Binary XOR
```python
(bitStr1^bitStr2).count(1)
```
