
# nifti_viewer/mask.py
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage.measure import marching_cubes
from scipy.ndimage import label, generate_binary_structure
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class SurfaceViewer3D(QWidget):
    """
    A QWidget that encapsulates a PyVista QtInteractor for rendering a 3D
    surface mesh extracted from a volume. The extraction is performed by:
      1. Using a fixed threshold (the volume's mean).
      2. Keeping only the largest connected component.
      3. Extracting the outer surface via the marching cubes algorithm.
      4. Smoothing and decimating the resulting mesh.
    Markers can also be overlaid as small spheres.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = None
        self.current_threshold = None
        self.mesh = None  # Cache the computed mesh
        self.layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("white")
    
    def set_volume(self, volume):
        """Set the volume data and compute the mesh using a fixed threshold."""
        # Ensure volume is a contiguous array and 3D.
        self.volume = np.ascontiguousarray(volume)
        if self.volume.ndim < 3:
            raise ValueError("Volume must be at least 3D.")
        self.current_threshold = np.mean(self.volume)
        self.mesh = None  # Reset any previously computed mesh.
        self.update_mesh(markers=[])
    
    def update_mesh(self, markers=None):
        """Update the displayed mesh and overlay any markers.
        
        Parameters:
            markers (list): A list of dictionaries (each with keys 'x', 'y', 'z', 'color')
                            to overlay as small spheres.
        """
        if self.volume is None:
            return
        
        if self.mesh is None:
            try:
                threshold = self.current_threshold
                # Create a binary mask using the fixed threshold.
                mask = self.volume > threshold

                # Keep only the largest connected component.
                structure = generate_binary_structure(3, 1)
                labeled, num_features = label(mask, structure=structure)
                if num_features > 0:
                    sizes = np.bincount(labeled.ravel())
                    if len(sizes) > 1:
                        largest_label = sizes[1:].argmax() + 1
                        mask = (labeled == largest_label)
                
                # Check that the mask has some True values.
                if not np.any(mask):
                    raise ValueError("No voxels above the threshold; check your data or threshold value.")
                
                # Extract the outer surface using marching cubes.
                verts, faces, normals, _ = marching_cubes(mask.astype(np.uint8), level=0.5)
                if faces.size == 0:
                    raise ValueError("No surface extracted (faces array is empty).")
                
                # PyVista expects faces to be in a flat array with a count prefix.
                faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
                mesh = pv.PolyData(verts, faces_pv)
                
                # For faster interactivity, smooth with fewer iterations and decimate.
                smoothed_mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1)
                decimated_mesh = smoothed_mesh.decimate(target_reduction=0.5)
                self.mesh = decimated_mesh

            except Exception as e:
                print("Error computing 3D mesh:", e)
                self.mesh = None
                return
        
        # Clear the current plot and add the mesh.
        self.plotter.clear()
        if self.mesh is not None:
            self.plotter.add_mesh(self.mesh, color='peachpuff', opacity=1, show_edges=True)
        
        # Overlay any markers as small spheres.
        if markers:
            for marker in markers:
                x = marker['x']
                y = marker['y']
                z = marker['z']
                sphere = pv.Sphere(radius=2.0, center=(x, y, z))
                self.plotter.add_mesh(sphere, color=marker['color'])
        
        self.plotter.reset_camera()

