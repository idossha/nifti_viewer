
# cli.py
import sys
import argparse
from PyQt5.QtWidgets import QApplication
from gui import MainWindow

def parse_args():
    parser = argparse.ArgumentParser(description="NIfTI Viewer with Fiducials")
    parser.add_argument("--nifti", type=str, help="Path to the NIfTI file to open")
    return parser.parse_args()

def run_cli():
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow()
    if args.nifti:
        window.load_nifti_from_path(args.nifti)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_cli()

