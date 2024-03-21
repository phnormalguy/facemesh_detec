import mediapipe as mp
import cv2 as cv
import numpy as np


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.4, minTrackCon=0.4):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.blue = 0
        self.green = 0
        self.red = 0
        self.color = (self.blue, self.green, self.red)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.img_show = cv.resize(self.imgRGB, (1920, 1080))
        self.results = self.faceMesh.process(self.imgRGB)
        self.liparr = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
                       321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
                       269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
                       14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81, ]
        self.right_Eyeblow = [276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336]
        self.brush = [143, 116, 123, 147, 213, 192, 214, 212, 216, 206, 203, 129, 209, 217, 114, 128, 233, 232, 231, 230, 229, 228, 31, 226, 35, 143]
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # self.mpDraw.draw_landmarks(self.img_show, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = self.img_show.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv.putText(self.img_show , str(id),(x,y),cv.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return self.img_show, faces

    def cropMouth(self, img, landmarks, color, alpha=0.6):

        points = np.array(landmarks)
        points = points.reshape((-1, 1, 2))
        mask = np.zeros_like(img)
        cv.fillPoly(mask, [points], self.color)

        return mask
    def cropMouth_White(self, img, landmarks, alpha=0.6):

        points = np.array(landmarks)
        points = points.reshape((-1, 1, 2))
        mask = np.zeros_like(img)
        cv.fillPoly(mask, [points], (255,255,255))

        return mask