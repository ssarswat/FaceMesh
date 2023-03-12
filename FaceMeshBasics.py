import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #We need to convert image from BGR to RGB for processing
    results = faceMesh.process(imgRGB)
    faces = []  # To recognize which face
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            face=[] #To store pixe values of a face
            for id, lm in enumerate (faceLms.landmark): #This will give id to landmarks for ease of recognition
                #print(lm) #This will print normalised values of the landmarks between 0 - 1
                ih, iw, ic = img.shape #Image Height, width, channels
                x,y = int(lm.x*iw), int(lm.y*ih) #For conversion to pixels
                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1) #For displaying id on facemesh
                print(id,x,y) #Will print landmark id with pixel
                face.append([x,y]) #Will append pixel values to list called face
            faces.append(face) #Will save pixel values belonging to each face to respective list

    if len(faces)!=0:
        print(len(faces))
    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)