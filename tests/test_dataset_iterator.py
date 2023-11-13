import pytest
import numpy as np

from mlrep.dataset_iterator import FromDataDataset


def test_pad():

    inp = np.array([[1,2,3,4],[1,2,3,4]]).T
    print(inp.shape)
    out = np.array([1,2,3,4])
    data = (inp,out)
    window_size = 3
    ite = FromDataDataset(data, window_size=window_size,skip_if_nan=True,pad=False,pad_value=0)

    out_e = []
    in_e = []
    for i,o in iter(ite):
        print("ite",i,o)
        in_e.append(i[window_size//2])
        out_e.append(o)
    in_e = np.vstack(in_e)
    
    assert( np.all(out_e == out[1:-1]))
    assert( np.all(in_e == inp[1:-1]))

    inp = np.array([[1,2,3,4],[1,2,3,4]]).T
    print(inp.shape)
    out = np.array([1,2,3,4])
    data = (inp,out)
    window_size = 3
    ite = FromDataDataset(data, window_size=window_size,skip_if_nan=True,pad=True,pad_value=0)

    out_e = []
    in_e = []
    for i,o in iter(ite):
        print("ite",i,o)
        in_e.append(i[window_size//2])
        out_e.append(o)
    in_e = np.vstack(in_e)
    
    assert( np.all(out_e == out))
    assert( np.all(in_e == inp))
def test_pad_nan():

    inp = np.array([[np.nan,2,3,4],[1,2,3,4]]).T
    print(inp.shape)
    out = np.array([1,2,3,4])
    data = (inp,out)
    window_size = 3
    ite = FromDataDataset(data, window_size=window_size,skip_if_nan=True,pad=True,pad_value=0)

    out_e = []
    in_e = []
    for i,o in iter(ite):
        print("ite",i,o)
        in_e.append(i[window_size//2])
        out_e.append(o)
    in_e = np.vstack(in_e)
    print(in_e)

    assert(np.sum(np.isnan(in_e))==0)
    assert(np.sum(np.isnan(out_e))==0)

def test_no_pad_nan():

    inp = np.array([[np.nan,2,3,4],[1,2,3,4]]).T
    print(inp.shape)
    out = np.array([1,2,3,4])
    data = (inp,out)
    window_size = 3
    ite = FromDataDataset(data, window_size=window_size,skip_if_nan=True,pad=False,pad_value=0)

    out_e = []
    in_e = []
    for i,o in iter(ite):
        print("ite",i,o)
        in_e.append(i[window_size//2])
        out_e.append(o)
    in_e = np.vstack(in_e)
    print(in_e)
    assert(np.sum(np.isnan(in_e))==0)
    assert(np.sum(np.isnan(out_e))==0)

    #assert( np.all(out_e == out))
    #assert( np.all(in_e == inp))


