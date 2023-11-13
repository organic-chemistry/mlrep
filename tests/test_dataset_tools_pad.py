import pytest
import numpy as np

from mlrep.dataset_tools import get_padded_view



def test_pad():
    arr= np.arange(2,4)

    res = get_padded_view(arr,index_slice=[0,2],pad_length=1,pad_value=0) 
    assert np.all(res==np.array([0,2]))
    res = get_padded_view(arr,index_slice=[len(arr)+2-2,len(arr)+2],pad_length=1,pad_value=0) 
    assert np.all(res==np.array([3,0]))

    res = get_padded_view(arr,index_slice=[len(arr)+2-2,len(arr)+2],pad_length=1,pad_value=1) 
    assert np.all(res==np.array([3,1]))

    res = get_padded_view(arr,index_slice=[0,len(arr)+2],pad_length=1,pad_value=0) 
    assert np.all(res==np.array([0,2,3,0]))

    res = get_padded_view(np.array([arr,arr]),index_slice=[0,len(arr)+2],pad_length=1,pad_value=0) 

    assert np.all(res==np.array([[0,0],[2,3],[2,3],[0,0]]))
    
    res = get_padded_view(np.array([arr,arr]),index_slice=[0,len(arr)+4],pad_length=2,pad_value=0) 

    assert np.all(res==np.array([[0,0],[0,0],[2,3],[2,3],[0,0],[0,0]]))