import cv2

vid = cv2.VideoCapture(0)
i=1
while(True):
    while(True):
        
        ret, frame = vid.read()
        cv2.imshow('frame', frame)

    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f"dataset\\jerome\image{i}.png",frame)
            i+=1
            break

vid.release()
cv2.destroyAllWindows()