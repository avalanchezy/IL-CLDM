"""
Dataset for PET image loading.

Used for AAE training (Stage 1).
For longitudinal prediction, use dataset_longitudinal.py.
"""

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import config


def read_list(file):
    """Read list of subject IDs from file."""
    with open(file, "r") as f:
        lines = f.read().split()
    return list(str(i) for i in lines)


def nifti_to_numpy(file):
    """Load NIfTI file and return numpy array."""
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data


class OneDataset(Dataset):
    """
    Dataset for single-timepoint PET images.
    Used for AAE training (Stage 1).
    """
    
    def __init__(self, root_Abeta=config.whole_Abeta, task=config.train, stage="train"):
        """
        Args:
            root_Abeta: Root directory containing PET images
            task: Path to file with subject IDs
            stage: 'train', 'validation', or 'test'
        """
        self.root_Abeta = root_Abeta
        self.task = task
        self.stage = stage
        
        if os.path.exists(task):
            self.images = read_list(task)
        else:
            # Auto-discover images if task file doesn't exist
            self.images = self._discover_images()
        
        self.length_dataset = len(self.images)
        self.len = len(self.images)

    def _discover_images(self):
        """Auto-discover images from directory."""
        images = []
        if os.path.exists(self.root_Abeta):
            for f in os.listdir(self.root_Abeta):
                if f.endswith('.nii.gz') or f.endswith('.nii'):
                    images.append(f)
        return images

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        name = self.images[index % self.len]
        path_Abeta = os.path.join(self.root_Abeta, name)
        
        Abeta = nifti_to_numpy(path_Abeta)
        
        return Abeta, name
