
# NIfTI Viewer

**NIfTI Viewer** is a Python application that allows users to load, view, and interact with NIfTI images. It provides both a graphical user interface (GUI) and a command-line interface (CLI).

## Features

- **2D Slice Viewing:**  
  Displays axial, coronal, and sagittal slices with correct orientation. The views are synchronized so that the crosshair always points to the same voxel in the volume.

- **3D Rendering:**  
  Displays a simple 3D mask rendering of the volume (with an adjustable threshold).

- **Fiducial Markers:**  
  Allows placing of fiducial markers (e.g., NAS, LPA, RPA) that are overlaid on the images and exported to JSON.

- **Command-Line Interface:**  
  Launch the viewer from the command line with a given NIfTI file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nifti_viewer.git
   cd nifti_viewer
