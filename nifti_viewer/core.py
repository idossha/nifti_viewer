
# core.py
import nibabel as nib
import numpy as np
import scipy.ndimage

def load_nifti(filepath):
    """
    Load a NIfTI file using nibabel.
    
    Returns:
      data (np.ndarray): Image data.
      header (nibabel header): Header information.
      affine (np.ndarray): 4x4 affine transformation.
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    return data, header, affine

def resample_to_isotropic(data, header):
    """
    Resample the volume to isotropic resolution if necessary.
    
    Parameters:
      data (np.ndarray): The image data.
      header: The NIfTI header.
    
    Returns:
      new_data (np.ndarray): The (possibly resampled) image data.
    """
    voxel_dims = header.get_zooms()[:3]
    if not np.allclose(voxel_dims, voxel_dims[0]):
        print("Voxel dimensions are not isotropic. Resampling to isotropic resolution.")
        min_dim = min(voxel_dims)
        scale_factors = [dim / min_dim for dim in voxel_dims]
        new_data = scipy.ndimage.zoom(data, zoom=scale_factors, order=1)
        return new_data
    else:
        return data

def load_and_preprocess(filepath):
    """
    Load a NIfTI file and resample it to isotropic resolution if necessary.
    
    Parameters:
      filepath (str): Path to the NIfTI file.
    
    Returns:
      data, header, affine
    """
    data, header, affine = load_nifti(filepath)
    data = resample_to_isotropic(data, header)
    return data, header, affine

def voxel_to_ras(voxel, affine):
    """
    Convert a voxel coordinate to RAS using the affine.
    
    Parameters:
      voxel (iterable): A 3-element iterable (i, j, k).
      affine (np.ndarray): 4x4 affine matrix.
    
    Returns:
      ras (np.ndarray): 3-element RAS coordinate.
    """
    v = np.array(list(voxel) + [1])
    ras = np.dot(affine, v)[:3]
    return ras

