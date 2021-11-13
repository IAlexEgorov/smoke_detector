import cv2 # Импорт модуля OpenCV

cap = cv2.VideoCapture("./videos/1.mp4"); 

cap.set(3,1280) # Установление длины окна
cap.set(4,700)  # Ширина окна

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      cv2.imshow('Frame',frame)
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
      break


cap.release()
cv2.destroyAllWindows()