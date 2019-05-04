# CSC-391_Project_Final
Augmented Reality Castle

The checkergrab.py program is used to obtain source data for camera calibration. Pressing 'p' will save the next frame that contains a checkerboard into the Data/Calibration/ directory, it's recommended that the user save about 12 frames with decent variation for effective camera calibration.

The camCalibration.py is basically a scaled down version of the cameraCalibration.py program we were provided. It reads in the output images from the checkergrab program, and outputs the camera calibration variables as matrixParams_mbp.npz, for use in the ARCastle.py program.

The ARCastle.py program is the main file for my project. It uses the matrixParams_mbp.npz file from camCalibration.py to project a castle onto a checkerboard based on the top, side, and front file names set on top of the program, these files must be in the Data/Textures/ directory.
