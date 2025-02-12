
# gui.py
import json
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QGridLayout, QRadioButton,
                             QButtonGroup, QGroupBox, QSlider, QLabel)
from PyQt5.QtCore import Qt

# Import core functions
from .core import load_and_preprocess, voxel_to_ras

# -----------------------------------------------------
# 2D Slice Viewer
# -----------------------------------------------------
class SliceViewer(FigureCanvas):
    def __init__(self, orientation='axial', parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.orientation = orientation  # 'axial', 'coronal', or 'sagittal'
        self.image = None
        self.crosshair_x = 0  # display coordinate (u)
        self.crosshair_y = 0  # display coordinate (v)
        self.marker_coords = []  # list of tuples (u, v, color)
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
            self.ax.imshow(self.image, cmap='gray', origin='lower')
        if self.view_label:
            self.ax.set_title(self.view_label)
        self.ax.axhline(self.crosshair_y, color='red', linestyle='--')
        self.ax.axvline(self.crosshair_x, color='red', linestyle='--')
        for (mx, my, mcolor) in self.marker_coords:
            self.ax.plot(mx, my, marker='o', color=mcolor, markersize=10)
        self.draw()


# -----------------------------------------------------
# 3D Mask Viewer
# -----------------------------------------------------
class SurfaceViewer3D(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.volume = None
        self.current_threshold = None

    def set_volume(self, volume, default_threshold=None):
        self.volume = volume
        if default_threshold is None:
            default_threshold = np.mean(volume)
        self.current_threshold = default_threshold
        self.update_mask(self.current_threshold, markers=[])
    
    def update_mask(self, threshold, markers=None):
        self.ax.clear()
        if self.volume is None:
            self.draw()
            return
        mask = self.volume > threshold
        coords = np.argwhere(mask)
        if coords.size > 0:
            max_points = 10000
            if coords.shape[0] > max_points:
                indices = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
                coords = coords[indices]
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
            self.ax.scatter(xs, ys, zs, c='lightgrey', marker='o', s=1, alpha=0.3)
        if markers:
            xs = [m['x'] for m in markers]
            ys = [m['y'] for m in markers]
            zs = [m['z'] for m in markers]
            colors = [m['color'] for m in markers]
            self.ax.scatter(xs, ys, zs, c=colors, s=50)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
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

        # Global crosshair in voxel coordinates (i, j, k) where i=row, j=column, k=slice.
        self.current_crosshair = [0, 0, 0]
        # List of fiducial markers; each marker is a dict with keys: 'x', 'y', 'z', 'type', 'color'
        self.markers = []
        # In this version we use three fiducial types.
        self.fiducial_options = [
            ("NAS", "red"),
            ("LPA", "green"),
            ("RPA", "blue")
        ]
        self.current_fiducial_name = self.fiducial_options[0][0]
        self.current_fiducial_color = self.fiducial_options[0][1]
        self.intensity_range = (0, 1)
        self.threeD_loaded = False

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
        marker_button = QPushButton("Place Marker")
        marker_button.clicked.connect(self.place_marker)
        button_layout.addWidget(marker_button)
        export_button = QPushButton("Export Fiducials")
        export_button.clicked.connect(self.export_fiducials)
        button_layout.addWidget(export_button)
        load3d_button = QPushButton("Load 3D Render")
        load3d_button.clicked.connect(self.load_3d_render)
        button_layout.addWidget(load3d_button)
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
        # Attach a unified mouse handler.
        self.axial_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)
        self.coronal_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)
        self.sagittal_viewer.mpl_connect('motion_notify_event', self.sync_crosshairs)
        grid_layout.addWidget(self.axial_viewer, 0, 0)
        grid_layout.addWidget(self.sagittal_viewer, 0, 1)
        grid_layout.addWidget(self.coronal_viewer, 1, 0)
        # 3D render panel.
        self.threeD_widget = QWidget()
        threeD_layout = QVBoxLayout(self.threeD_widget)
        self.threeD_placeholder = QLabel("3D render not loaded.\nClick 'Load 3D Render' to load.")
        self.threeD_placeholder.setAlignment(Qt.AlignCenter)
        threeD_layout.addWidget(self.threeD_placeholder)
        self.viewer_3d = SurfaceViewer3D()
        self.viewer_3d.hide()
        threeD_layout.addWidget(self.viewer_3d)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.hide()
        threeD_layout.addWidget(QLabel("Mask Threshold"))
        threeD_layout.addWidget(self.slider)
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
            self.slider.setValue(500)
            self.update_slice_views()
            self.update_coord_display()

    def load_nifti_from_path(self, filepath):
        """Load a NIfTI file (without file dialog) for CLI use."""
        data, header, affine = load_and_preprocess(filepath)
        self.nifti_data = data
        self.nifti_header = header
        self.affine = affine
        dims = self.nifti_data.shape
        self.current_crosshair = [dims[0] // 2, dims[1] // 2, dims[2] // 2]
        self.intensity_range = (float(np.min(self.nifti_data)), float(np.max(self.nifti_data)))
        self.slider.setValue(500)
        self.update_slice_views()
        self.update_coord_display()

    def load_3d_render(self):
        if self.nifti_data is None:
            return
        default_thresh = np.mean(self.nifti_data)
        self.viewer_3d.set_volume(self.nifti_data, default_threshold=default_thresh)
        self.viewer_3d.update_mask(default_thresh, markers=self.markers)
        self.slider.setValue(500)
        self.threeD_placeholder.hide()
        self.viewer_3d.show()
        self.slider.show()
        self.threeD_loaded = True

    def update_slice_views(self):
        if self.nifti_data is None:
            return
        # Global voxel coordinate: (i, j, k) with i=row, j=column, k=slice
        i, j, k = self.current_crosshair
        dims = self.nifti_data.shape  # (X, Y, Z)

        # Extract raw slices.
        axial_slice = self.nifti_data[:, :, int(round(k))]      # shape (X, Y)
        coronal_slice = self.nifti_data[:, int(round(j)), :]     # shape (X, Z)
        sagittal_slice = self.nifti_data[int(round(i)), :, :]     # shape (Y, Z)

        # For orientation fixes, we apply:
        # Axial: flip left-right so that left appears on the right.
        axial_disp = np.fliplr(axial_slice)
        # Coronal: flip vertically so that anterior is on top.
        coronal_disp = np.flipud(coronal_slice)
        # Sagittal: flip vertically so that inferior is at bottom.
        sagittal_disp = np.flipud(sagittal_slice)

        # For markers, we adjust accordingly.
        # Axial view: originally, global (i, j) should be displayed at (j, i).
        # After fliplr, horizontal coordinate becomes: (Y - 1 - j)
        Y_axial = axial_disp.shape[1]
        axial_markers = []
        for m in self.markers:
            if int(round(m['z'])) == int(round(k)):
                # Original marker global coordinate: (i, j)
                # Display coordinate: (Y_axial - 1 - j, i)
                axial_markers.append((Y_axial - 1 - m['y'], m['x'], m['color']))
        
        # Coronal view: slice shape is (X, Z). We flip vertically.
        X_coronal = coronal_disp.shape[0]
        coronal_markers = []
        for m in self.markers:
            if int(round(m['y'])) == int(round(j)):
                # Global (i, k) should display as (k, i) originally.
                # After flipud, vertical coordinate becomes: (X_coronal - 1 - i)
                coronal_markers.append((m['z'], X_coronal - 1 - m['x'], m['color']))
        
        # Sagittal view: slice shape is (Y, Z). We flip vertically.
        Y_sagittal = sagittal_disp.shape[0]
        sagittal_markers = []
        for m in self.markers:
            if int(round(m['x'])) == int(round(i)):
                # Global (j, k) should display as (k, j) originally.
                # After flipud, vertical coordinate becomes: (Y_sagittal - 1 - j)
                sagittal_markers.append((m['z'], Y_sagittal - 1 - m['y'], m['color']))

        self.axial_viewer.set_image(axial_disp, markers=axial_markers)
        self.coronal_viewer.set_image(coronal_disp, markers=coronal_markers)
        self.sagittal_viewer.set_image(sagittal_disp, markers=sagittal_markers)

        # Update crosshairs.
        # Axial: global (i, j) → display: (Y_axial - 1 - j, i)
        self.axial_viewer.update_crosshair(Y_axial - 1 - j, i)
        # Coronal: global (i, k) → display: (k, X_coronal - 1 - i)
        self.coronal_viewer.update_crosshair(k, X_coronal - 1 - i)
        # Sagittal: global (j, k) → display: (k, Y_sagittal - 1 - j)
        self.sagittal_viewer.update_crosshair(k, Y_sagittal - 1 - j)

        if self.threeD_loaded:
            self.viewer_3d.update_mask(self.viewer_3d.current_threshold, markers=self.markers)
        self.update_coord_display()

    def sync_crosshairs(self, event):
        if event.button is None or event.xdata is None or event.ydata is None:
            return
        dims = self.nifti_data.shape
        # Because we are displaying with flips, we must invert the mapping.
        if event.inaxes == self.axial_viewer.ax:
            # Axial display: we showed crosshair at (Y_axial - 1 - j, i).
            # Let Y_axial = image width of axial view.
            Y_axial = self.axial_viewer.image.shape[1]
            # Inversion: if user clicks at (u, v) then:
            # global j = Y_axial - 1 - u, global i = v.
            new_i = int(round(v := event.ydata))
            new_j = int(round(Y_axial - 1 - event.xdata))
            self.current_crosshair[0] = new_i
            self.current_crosshair[1] = new_j
        elif event.inaxes == self.coronal_viewer.ax:
            # Coronal display: crosshair was shown at (k, X_coronal - 1 - i),
            # where X_coronal = height of the coronal image.
            X_coronal = self.coronal_viewer.image.shape[0]
            new_i = int(round(X_coronal - 1 - event.ydata))
            new_k = int(round(event.xdata))
            self.current_crosshair[0] = new_i
            self.current_crosshair[2] = new_k
        elif event.inaxes == self.sagittal_viewer.ax:
            # Sagittal display: crosshair shown at (k, Y_sagittal - 1 - j)
            Y_sagittal = self.sagittal_viewer.image.shape[0]
            new_j = int(round(Y_sagittal - 1 - event.ydata))
            new_k = int(round(event.xdata))
            self.current_crosshair[1] = new_j
            self.current_crosshair[2] = new_k
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
        # Replace any existing marker of the selected type.
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

    def on_slider_changed(self, slider_value):
        if self.nifti_data is None or not self.threeD_loaded:
            return
        min_val, max_val = self.intensity_range
        thresh = min_val + (slider_value / 1000.0) * (max_val - min_val)
        self.viewer_3d.current_threshold = thresh
        self.viewer_3d.update_mask(thresh, markers=self.markers)

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

