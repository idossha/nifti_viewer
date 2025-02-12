
# nifti_viewer/gui.py
import json
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QGridLayout, QRadioButton,
                             QButtonGroup, QGroupBox, QLabel)
from PyQt5.QtCore import Qt

# Import core functions.
from .core import load_and_preprocess, voxel_to_ras

# Import 3D viewer from our separate module.
from .mask import SurfaceViewer3D

def rotate180_coords(x, y, shape):
    """
    Given an (x, y) coordinate in a 2D array (shape = (height, width)),
    return the coordinate after a 180-degree rotation (flip LR + flip UD).
    """
    h, w = shape
    return (w - 1 - x, h - 1 - y)

def rotate180_coords_inverse(x, y, shape):
    """
    The inverse transform is identical for a 180-degree rotation.
    """
    return rotate180_coords(x, y, shape)

# -----------------------------------------------------
# 2D Slice Viewer
# -----------------------------------------------------
class SliceViewer(FigureCanvas):
    def __init__(self, orientation='axial', parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.orientation = orientation  # 'axial', 'coronal', or 'sagittal'
        self.image = None
        self.crosshair_x = 0  # display coordinate (horizontal)
        self.crosshair_y = 0  # display coordinate (vertical)
        self.marker_coords = []  # list of tuples (x, y, color)
        self.view_label = ""  # Title for the view.
    
    def set_image(self, image, markers=None):
        """Display a 2D numpy array with the specified markers."""
        self.image = image
        self.marker_coords = markers if markers is not None else []
        self.redraw()
    
    def update_crosshair(self, x, y):
        """Update the crosshair position in display coordinates and redraw."""
        self.crosshair_x = int(round(x))
        self.crosshair_y = int(round(y))
        self.redraw()
    
    def redraw(self):
        """Clear and redraw the image, crosshair, markers, and title."""
        self.ax.clear()
        if self.image is not None:
            # We'll use origin='upper' so that array row 0 is top of the display.
            self.ax.imshow(self.image, cmap='gray', origin='upper')
        if self.view_label:
            self.ax.set_title(self.view_label)
        # Draw crosshair lines
        self.ax.axhline(self.crosshair_y, color='red', linestyle='--', linewidth=0.5)
        self.ax.axvline(self.crosshair_x, color='red', linestyle='--', linewidth=0.5)
        # Draw markers
        for (mx, my, mcolor) in self.marker_coords:
            self.ax.plot(mx, my, marker='o', color=mcolor, markersize=5)
        self.draw()

# -----------------------------------------------------
# Main Application Window
# -----------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI Mask Viewer with Fiducials (3D & 2D)")
        self.nifti_data = None
        self.nifti_header = None
        self.affine = None  # For voxel-to-RAS conversion.

        # Global crosshair in voxel coordinates (i, j, k) = (X, Y, Z).
        # We'll interpret:
        #   i = left-right,
        #   j = anterior-posterior,
        #   k = inferior-superior.
        self.current_crosshair = [0, 0, 0]

        # List of fiducial markers; each marker is a dict with keys:
        #   'x', 'y', 'z', 'type', 'color'
        self.markers = []

        # Fiducial types.
        self.fiducial_options = [
            ("NAS", "red"),
            ("LPA", "green"),
            ("RPA", "blue")
        ]
        self.current_fiducial_name = self.fiducial_options[0][0]
        self.current_fiducial_color = self.fiducial_options[0][1]
        self.intensity_range = (0, 1)
        self.threeD_loaded = False

        # Set focus to capture key events.
        self.setFocusPolicy(Qt.StrongFocus)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top button bar.
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load NIfTI")
        load_button.clicked.connect(self.load_nifti)
        button_layout.addWidget(load_button)
        export_button = QPushButton("Export Fiducials")
        export_button.clicked.connect(self.export_fiducials)
        button_layout.addWidget(export_button)
        main_layout.addLayout(button_layout)

        # Fiducial selection.
        fiducial_group = QGroupBox("Fiducial Types")
        fiducial_layout = QHBoxLayout()
        self.fiducial_button_group = QButtonGroup()
        for fid_name, fid_color in self.fiducial_options:
            radio = QRadioButton(fid_name)
            if fid_name == self.current_fiducial_name:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, name=fid_name, color=fid_color:
                                  self.on_fiducial_selected(checked, name, color))
            self.fiducial_button_group.addButton(radio)
            fiducial_layout.addWidget(radio)
        fiducial_group.setLayout(fiducial_layout)
        main_layout.addWidget(fiducial_group)

        # 2×2 grid for 2D and 3D views.
        grid_layout = QGridLayout()
        self.axial_viewer = SliceViewer(orientation='axial')
        self.coronal_viewer = SliceViewer(orientation='coronal')
        self.sagittal_viewer = SliceViewer(orientation='sagittal')
        self.axial_viewer.view_label = "Axial"
        self.coronal_viewer.view_label = "Coronal"
        self.sagittal_viewer.view_label = "Sagittal"

        # Attach a unified mouse handler to sync crosshairs on mouse move.
        self.axial_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)
        self.coronal_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)
        self.sagittal_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)

        grid_layout.addWidget(self.axial_viewer, 0, 0)
        grid_layout.addWidget(self.sagittal_viewer, 0, 1)
        grid_layout.addWidget(self.coronal_viewer, 1, 0)

        # 3D render panel.
        self.threeD_widget = QWidget()
        threeD_layout = QVBoxLayout(self.threeD_widget)
        self.threeD_placeholder = QLabel("3D render not loaded.")
        self.threeD_placeholder.setAlignment(Qt.AlignCenter)
        threeD_layout.addWidget(self.threeD_placeholder)
        self.viewer_3d = SurfaceViewer3D()
        self.viewer_3d.hide()
        threeD_layout.addWidget(self.viewer_3d)
        grid_layout.addWidget(self.threeD_widget, 1, 1)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        main_layout.addLayout(grid_layout)

        # Coordinates display panel.
        coord_group = QGroupBox("Coordinates")
        coord_layout = QVBoxLayout()
        self.coord_crosshair_label = QLabel("Crosshair: RAS: N/A / Voxel: N/A")
        coord_layout.addWidget(self.coord_crosshair_label)
        self.coord_fiducial_labels = {}
        for fid_name, _ in self.fiducial_options:
            lbl = QLabel(f"{fid_name}: RAS: N/A / Voxel: N/A")
            self.coord_fiducial_labels[fid_name] = lbl
            coord_layout.addWidget(lbl)
        coord_group.setLayout(coord_layout)
        main_layout.addWidget(coord_group)

    def keyPressEvent(self, event):
        # If the "m" key is pressed, place a marker at the current crosshair.
        if event.key() == Qt.Key_M:
            self.place_marker()
        else:
            super().keyPressEvent(event)

    def on_fiducial_selected(self, checked, name, color):
        if checked:
            self.current_fiducial_name = name
            self.current_fiducial_color = color

    def load_nifti(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if filename:
            data, header, affine = load_and_preprocess(filename)
            self.nifti_data = data
            self.nifti_header = header
            self.affine = affine
            dims = self.nifti_data.shape  # (X, Y, Z)
            self.current_crosshair = [dims[0] // 2, dims[1] // 2, dims[2] // 2]
            self.intensity_range = (float(np.min(self.nifti_data)), float(np.max(self.nifti_data)))
            self.update_slice_views()
            self.update_coord_display()
            # Automatically load the 3D render.
            self.load_3d_render()

    def load_nifti_from_path(self, filepath):
        """Load a NIfTI file (without file dialog) for CLI use."""
        data, header, affine = load_and_preprocess(filepath)
        self.nifti_data = data
        self.nifti_header = header
        self.affine = affine
        dims = self.nifti_data.shape
        self.current_crosshair = [dims[0] // 2, dims[1] // 2, dims[2] // 2]
        self.intensity_range = (float(np.min(self.nifti_data)), float(np.max(self.nifti_data)))
        self.update_slice_views()
        self.update_coord_display()
        self.load_3d_render()

    def load_3d_render(self):
        if self.nifti_data is None:
            return
        # Set the volume in the 3D viewer. The mesh is computed with a fixed threshold.
        self.viewer_3d.set_volume(self.nifti_data)
        # Overlay any existing markers.
        self.viewer_3d.update_mesh(markers=self.markers)
        self.threeD_placeholder.hide()
        self.viewer_3d.show()
        self.threeD_loaded = True

    def update_slice_views(self):
        if self.nifti_data is None:
            return
        i, j, k = self.current_crosshair
        # (i, j, k) = (X, Y, Z)

        # --------------------------------------------------------
        # 1) Extract raw slices for each plane
        #    - Axial:    fix k; shape (X, Y) => (i, j)
        #    - Coronal:  fix j; shape (X, Z) => (i, k)
        #    - Sagittal: fix i; shape (Y, Z) => (j, k)
        # --------------------------------------------------------
        axial_slice    = self.nifti_data[:, :, k]   # shape (X, Y) => (i, j)
        coronal_slice  = self.nifti_data[:, j, :]   # shape (X, Z) => (i, k)
        sagittal_slice = self.nifti_data[i, :, :]   # shape (Y, Z) => (j, k)

        # --------------------------------------------------------
        # 2) Transpose => so dimension 0 = 'height', dimension 1 = 'width'
        #    Then rotate 180°.
        #    After transpose:
        #      Axial   => shape (j, i)
        #      Coronal => shape (k, i)
        #      Sagittal=> shape (k, j)
        # --------------------------------------------------------
        axial_disp    = np.flipud(np.fliplr(axial_slice.T))
        coronal_disp  = np.flipud(np.fliplr(coronal_slice.T))
        sagittal_disp = np.flipud(np.fliplr(sagittal_slice.T))

        # Store final shapes for crosshair/marker coordinate transforms
        axial_shape    = axial_disp.shape   # (j, i)
        coronal_shape  = coronal_disp.shape # (k, i)
        sagittal_shape = sagittal_disp.shape# (k, j)

        # --------------------------------------------------------
        # 3) Markers: each marker has voxel coords (mx, my, mz).
        #    We only draw it if it lies on the slice plane.
        #    Then we map to (x, y) in the 2D display, then rotate 180°.
        # --------------------------------------------------------
        axial_markers    = []
        coronal_markers  = []
        sagittal_markers = []
        for m in self.markers:
            mx = m['x']
            my = m['y']
            mz = m['z']
            color = m['color']
            # Axial => fixed k => if round(mz) == k
            if int(round(mz)) == int(round(k)):
                # Before flipping, the transposed coordinate is (x = i, y = j) => (mx, my).
                # Then apply rotate180_coords.
                tx, ty = rotate180_coords(mx, my, axial_shape)
                axial_markers.append((tx, ty, color))
            # Coronal => fixed j => if round(my) == j
            if int(round(my)) == int(round(j)):
                # Transposed => (x = i, y = k) => (mx, mz)
                tx, ty = rotate180_coords(mx, mz, coronal_shape)
                coronal_markers.append((tx, ty, color))
            # Sagittal => fixed i => if round(mx) == i
            if int(round(mx)) == int(round(i)):
                # Transposed => (x = j, y = k) => (my, mz)
                tx, ty = rotate180_coords(my, mz, sagittal_shape)
                sagittal_markers.append((tx, ty, color))

        # --------------------------------------------------------
        # 4) Set images & markers
        # --------------------------------------------------------
        self.axial_viewer.set_image(axial_disp, markers=axial_markers)
        self.coronal_viewer.set_image(coronal_disp, markers=coronal_markers)
        self.sagittal_viewer.set_image(sagittal_disp, markers=sagittal_markers)

        # --------------------------------------------------------
        # 5) Crosshair in display coords
        #    Axial:   (x= i, y= j)
        #    Coronal: (x= i, y= k)
        #    Sagittal:(x= j, y= k)
        #    Then apply rotate180_coords before calling update_crosshair().
        # --------------------------------------------------------
        ax_x, ax_y = rotate180_coords(i, j, axial_shape)
        self.axial_viewer.update_crosshair(ax_x, ax_y)

        co_x, co_y = rotate180_coords(i, k, coronal_shape)
        self.coronal_viewer.update_crosshair(co_x, co_y)

        sa_x, sa_y = rotate180_coords(j, k, sagittal_shape)
        self.sagittal_viewer.update_crosshair(sa_x, sa_y)

        self.update_coord_display()

    def sync_crosshairs(self, event):
        """
        When the user moves the mouse in a given slice viewer, we invert
        the transform to get the new (i, j, k). That way, crosshair
        updates remain consistent across all views.
        """
        if event.button is None or event.xdata is None or event.ydata is None:
            return
        i, j, k = self.current_crosshair

        # Axial => final shape is (j, i), so display.x => i, display.y => j.
        # Coronal => shape (k, i), so display.x => i, display.y => k.
        # Sagittal => shape (k, j), so display.x => j, display.y => k.

        if event.inaxes == self.axial_viewer.ax:
            axial_shape = self.axial_viewer.image.shape  # (j, i)
            # invert the 180° flip
            trans_i, trans_j = rotate180_coords_inverse(int(round(event.xdata)),
                                                        int(round(event.ydata)),
                                                        axial_shape)
            new_i = trans_i
            new_j = trans_j
            self.current_crosshair = [new_i, new_j, k]

        elif event.inaxes == self.coronal_viewer.ax:
            coronal_shape = self.coronal_viewer.image.shape  # (k, i)
            trans_i, trans_k = rotate180_coords_inverse(int(round(event.xdata)),
                                                        int(round(event.ydata)),
                                                        coronal_shape)
            new_i = trans_i
            new_k = trans_k
            self.current_crosshair = [new_i, j, new_k]

        elif event.inaxes == self.sagittal_viewer.ax:
            sagittal_shape = self.sagittal_viewer.image.shape  # (k, j)
            trans_j, trans_k = rotate180_coords_inverse(int(round(event.xdata)),
                                                        int(round(event.ydata)),
                                                        sagittal_shape)
            new_j = trans_j
            new_k = trans_k
            self.current_crosshair = [i, new_j, new_k]

        self.update_slice_views()

    def voxel_to_ras(self, voxel):
        if self.affine is None:
            return None
        return voxel_to_ras(voxel, self.affine)

    def update_coord_display(self):
        def fmt(coord):
            return f"({coord[0]}, {coord[1]}, {coord[2]})"
        cross_vox = self.current_crosshair
        cross_ras = self.voxel_to_ras(cross_vox)
        self.coord_crosshair_label.setText(
            f"Crosshair: RAS: {fmt(cross_ras) if cross_ras is not None else 'N/A'} / Voxel: {fmt(cross_vox)}"
        )
        latest = {fid: None for fid, _ in self.fiducial_options}
        for marker in self.markers:
            latest[marker.get('type')] = marker
        for fid, label in self.coord_fiducial_labels.items():
            marker = latest.get(fid)
            if marker is not None:
                vox = (marker['x'], marker['y'], marker['z'])
                ras = self.voxel_to_ras(vox)
                label.setText(f"{fid}: RAS: {fmt(ras)} / Voxel: {fmt(vox)}")
            else:
                label.setText(f"{fid}: RAS: N/A / Voxel: N/A")

    def place_marker(self):
        """
        Place a 3D marker at the current crosshair voxel coords.
        (Invoked when the user presses the "m" key.)
        """
        # Remove any existing marker of the selected type.
        self.markers = [m for m in self.markers if m.get('type') != self.current_fiducial_name]
        marker = {
            'x': self.current_crosshair[0],
            'y': self.current_crosshair[1],
            'z': self.current_crosshair[2],
            'type': self.current_fiducial_name,
            'color': self.current_fiducial_color
        }
        self.markers.append(marker)
        self.update_slice_views()
        self.update_coord_display()
        # Update 3D view with new marker.
        if self.threeD_loaded:
            self.viewer_3d.update_mesh(markers=self.markers)

    def export_fiducials(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Fiducials", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.markers, f, indent=4)
            print("Exported fiducials to", filename)


def main():
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

