import numpy as np
import tensorflow as tf
import cv2
import glob
import matplotlib.pyplot as plt
from skimage import morphology
from collections import deque

class LaneDetectImage:
    def display(self, img, title, color = 1):
        if color:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap = 'gray')

        plt.title(title)
        plt.axis('off')
        plt.show()

    def CameraCalibrations(self, folder, nx, ny, choice):
        objpoints = [] #3D
        imgpoints = []
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        assert len(folder) != 0
        for fname in folder:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCornersSB(gray, (nx, ny))
            img_sz = gray.shape[::-1]
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
                if choice:
                    draw_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    self.display(draw_corners, 'Found all corners:{}'.format(ret))

        if len(objpoints) == len(imgpoints) and len(objpoints) != 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_sz, None, None)
            return {'ret': ret, 'cameraMatrix': mtx, 'distorsionCoeff': dist, \
                    'rotationVec': rvecs, 'translationVec': tvecs}

        else:
            raise cv2.Error("Camera Calibration Failed!")


    def correction(self, image, calib_parms, showMe = 0):
        corrected = cv2.undistort(image, calib_parms['cameraMatrix'], calib_parms['distorsionCoeff'], \
                                    None, calib_parms['cameraMatrix'])
        if showMe:
            self.display(image, 'original', color = 1)
            self.display(corrected, 'After correction', color=1)
        return corrected

















