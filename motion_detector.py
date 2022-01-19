import cv2, time
from datetime import datetime
import pandas

first_frame=None #initalization of the frame
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0) #1 external bir camera olabilir

while True:
    check, frame=video.read() #check video çekiyor mu, frame videodaki framelaer
    status=0
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray
    gray=cv2.GaussianBlur(gray,(21,21),0) #to reduce noise

    if first_frame is None:
        first_frame=gray #to capture first frame(background)
        continue # for the first loop the rest of the code is not necessary

    delta_frame=cv2.absdiff(first_frame, gray) #detect moving object
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1] # IF the difference >30 than object is moving, assign white to the moving pixel
    thresh_delta=cv2.erode(thresh_delta, None, iterations=5)
    thresh_delta=cv2.dilate(thresh_delta, None, iterations=5) #dilatation

    (cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find Countours of the object

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue #continue to the beginning of the for loop(next contour)
        status=1
        (x,y,w,h)=cv2.boundingRect(contour) #boundingRect takes the x,y,w and h of the countour
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:] #memory'i meşgul etmemek için

    if status_list[-1]==1 and status_list[-2]==0: #record time that object enters
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1: #record time that object leaves
        times.append(datetime.now())

    cv2.imshow("Frame" ,frame)
    cv2.imshow("Gray Frame" ,gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_delta)

    key=cv2.waitKey(1) #wait 1ms and close the previous image(show next image)
    #print(gray)
    #print(delta_frame)

    if key==ord('q'): #if operator presses q break the loop
        if status==1:
            times.append(datetime.now()) #kapattığımızda da çıkış olarak kaydetmesi için
        break
    #print(status)

print(status_list)
print(times)

for i in range(0,len(times),2): # start time-end time olarak ilerlediği için start'lara bakarak iterate ediyoruz
    df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)
df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows
