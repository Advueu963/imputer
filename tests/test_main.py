"""A simple test to ensure the test framework is working."""
import pytest 
from imputer.baseline_imputer import BaselineImputer, ImputeMode
import numpy as np



#1d test data
large_n = 24000 
rng = np.random.RandomState(2023)
point_np = rng.rand(large_n).astype(np.float32)
reference = rng.rand(large_n).astype(np.float32)
ref_1d_dataset = rng.rand(large_n, 200).astype(np.float32)
colations = np.zeros(large_n, dtype=bool)
colations[1::3] = True

#2d test data
n_samples, n_features = 300, 10
point_np_2d = rng.rand(n_samples, n_features).astype(np.float32)
reference_2d = rng.rand(n_samples, n_features).astype(np.float32)
ref_2d_dataset = rng.rand(n_samples, n_features, 100).astype(np.float32)
colations_2d = np.zeros((n_samples, n_features), dtype=bool)
colations_2d[:, 1::3] = True


#image

h,w, channels = 32,32,3
image = rng.rand(h,w,channels).astype(np.float32)
reference_img = rng.rand(h,w,channels).astype(np.float32)
img_dataset = rng.rand(h,w,channels, 50).astype(np.float32)
colations_img = np.zeros((h,w), dtype=bool)
colations_img[::2, ::2] = True


def run_impute_test(p, r, c, point, ref, ref_data, col):
    
    point_np = np.asarray(point)
    ref_np = np.asarray(ref)
    ref_data_np = np.asarray(ref_data)
    mask_np = np.asarray(col, dtype=bool)

    #static

    imputer_static = BaselineImputer(reference_data=ref_np, mode=ImputeMode.STATIC)
    out_static = imputer_static.impute(p, c)
    expected_static = point_np.copy()
    expected_static[mask_np] = ref_np[mask_np]
    assert np.allclose(np.asarray(out_static), expected_static, rtol=0, atol=0)

    #mean

    imputer_mean = BaselineImputer(reference_data=ref_data_np, mode=ImputeMode.MEAN)
    out_mean = imputer_mean.impute(p, c)
    mean_vec = np.mean(ref_data_np, axis=-1)
    expected_mean = point_np.copy()
    expected_mean[mask_np] = mean_vec[mask_np]
    assert np.allclose(np.asarray(out_mean), expected_mean, rtol=0, atol=0)

    #median
    imputer_med = BaselineImputer(reference_data=ref_data_np, mode=ImputeMode.MEDIAN)
    out_med = imputer_med.imputeimpute(p, c)
    median_vec = np.median(ref_data_np, axis=-1)
    expected_med = point_np.copy()
    expected_med[mask_np] = median_vec[mask_np]
    assert np.allclose(np.asarray(out_med), expected_med, rtol=0, atol=0)

    return True, True, True
     

def test_main():
    assert True


def test_impute_1d_jax():

    import jax.numpy as jnp 

    p = jnp.array(point_np)
    r = jnp.array(reference)
    c = jnp.array(colations, dtype= bool)

    run_impute_test(p,r,c, point_np, reference,ref_1d_dataset, colations)

def test_impute_2d_jax():

    import jax.numpy as jnp 

    p = jnp.array(point_np_2d)
    r = jnp.array(reference_2d)
    c = jnp.array(colations_2d)

    run_impute_test(p,r,c, point_np_2d, reference_2d,ref_2d_dataset, colations_2d)



def test_impute_1d_tensor():

    import torch 

    p = torch.tensor(point_np)
    r = torch.tensor(reference)
    c = torch.tensor(colations, dtype=torch.bool)

    run_impute_test(p,r,c, point_np, reference,ref_1d_dataset, colations)

def test_impute_2d_tensor():

    import torch 
    p = torch.tensor(point_np_2d)
    r = torch.tensor(reference_2d)
    c = torch.tensor(colations_2d)

    run_impute_test(p,r,c, point_np_2d, reference_2d,ref_2d_dataset, colations_2d)



def test_impute_image():


    p = image
    r = reference_img
    c = colations_img
    
    run_impute_test(p,r,c, image, reference_img, img_dataset, colations_img) 
    







