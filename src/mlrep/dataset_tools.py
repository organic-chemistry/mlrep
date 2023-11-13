
import torch
import random
import numpy as np

class RandomFileIterator(torch.utils.data.IterableDataset):
    """
    This class allow to randomly swhich between the differents iterators 
    """
    def __init__(self, iterators, max_length=None):
        self.iterators = iterators
        self.max_length = max_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_length is not None and len(self) >= self.max_length:
            raise StopIteration
        random_iterator = random.choice(self.iterators)
        try:
            return next(random_iterator)
        except StopIteration:
            self.iterators.remove(random_iterator)
            if not self.iterators:
                raise StopIteration
            return next(self)

    def __len__(self):
        if self.max_length is None:
            return sum(1 for _ in self.iterators)
        else:
            return self.max_length
        

class MyIterableDataset(torch.utils.data.IterableDataset):
     """
     Helper class to create an iterator at each period
     """
     def __init__(self, list_dataset):
         super(MyIterableDataset).__init__()
         self.list_dataset = list_dataset
     def __iter__(self):
        return RandomFileIterator([iter(ds) for ds in self.list_dataset])
     

class PaddedArrayView:
    #Array to avoid the explicit concatanation
    # return the index on the array [0]*pad + arr + [0] * pad
    def __init__(self, arr, pad_length,pad_value=0):
        self.arr = arr
        self.pad_shape = self.arr.shape
        self.pad_dim = len(self.arr.shape)
        self.pad_length = pad_length
        self.len=len(arr)
        self.pad_value = pad_value

    def get_view(self, index_slice):
        
        start, end = index_slice
        
        assert(end<= self.len + 2* self.pad_length)

        
        start_index = max(0, start - self.pad_length)
        end_index = min(len(self.arr), end - self.pad_length)
        
        # Create a padded slice.
        ret = self.arr[start_index:end_index] 

        extra_l=max(0, self.pad_length-start) 
        if extra_l>0 :
            if self.pad_dim == 1:
                left = np.zeros(extra_l)
            else:
                left = np.zeros([extra_l]+list(self.pad_shape)[1:])
            
            ret = np.concatenate([left+self.pad_value,ret])

        extra_r=max(0, end - self.pad_length-self.len)
        if  extra_r>0 :
            if self.pad_dim == 1:
                right = np.zeros(extra_r)
            else:
                right = np.zeros([extra_r]+list(self.pad_shape)[1:])
            
            ret = np.concatenate([ret,right+self.pad_value])
        
        	
      
        
        return ret
    

def get_padded_view(arr,index_slice,pad_length,pad_value=0):
    
    start, end = index_slice
    
    len_arr = len(arr)

    if start < 0 or end < 0 :
        print("Negative index not supported")
        raise

    assert(end<= len_arr + 2* pad_length)
    
    
    start_index = max(0, start - pad_length)
    end_index = min(len_arr, end - pad_length)
    
    #Nedd to pad left or right
    extra_l=max(0, pad_length-start) 
    extra_r=max(0, end - pad_length-len_arr)

    if extra_l == 0 and extra_r == 0:
        return arr[start_index:end_index]

    ret = arr[start_index:end_index] 
    pad_dim = len(arr.shape)
    pad_shape = list(arr.shape)
    dtype=arr.dtype

    if extra_l>0 :
        if pad_dim == 1:
            left = np.zeros(extra_l,dtype=dtype)
        else:
            left = np.zeros([extra_l]+pad_shape[1:],dtype=dtype)
        
        ret = np.concatenate([left+pad_value,ret])

    if  extra_r>0 :
        if pad_dim == 1:
            right = np.zeros(extra_r,dtype=dtype)
        else:
            right = np.zeros([extra_r]+pad_shape[1:],dtype=dtype)
        
        ret = np.concatenate([ret,right+pad_value],dtype=dtype)
    
    
    return ret

"""
arr = np.array([[1, 2],[ 3, 4]])
pad_length = 3
index_slice = (0, 6)
result = get_padded_view(arr,index_slice,pad_length,pad_value=0)
print(result)
"""

"""
# Example usage:
arr = np.array([[1, 2],[ 3, 4]])
pad_length = 3
index_slice = (0, 4)
padded_array = PaddedArrayView(arr, pad_length,pad_value=1)
result = padded_array.get_view(index_slice)
print(result)
"""