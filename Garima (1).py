import face_recognition
import cv2
import os
import pandas as pd
from datetime import datetime
import time
from tkinter import *
import cvlib as cv
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace
from tensorflow.keras.preprocessing.image import img_to_array

lis=[]

face_cascade = cv2.CascadeClassifier(r'/home/sageiac/Downloads/face_dectect.xml')
na=""

count=0
model1=load_model(r'/home/sageiac/Desktop/liveness.model')

def gender(frame):
    padding = 20
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):        
        (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
        (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
        face_crop = np.copy(frame[startY:endY, startX:endX]) 
        (label, confidence) = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        label = label[::-1][idx]
        label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
       
        return label
def name():
   
    
     
    root = Tk()
    root.title("Credintials Login")
    root.geometry("500x280")
    root.geometry()


    root.minsize(200, 200)

    # set maximum window size value
    root.maxsize(500, 200)
    c=Canvas(root,bg="gray16",height=200,width=200)
    filename=PhotoImage(file=r'/home/sageiac/Downloads/fw-bg-gradient.png')
    background_lable=Label(root,image=filename)
    background_lable.place(x=0,y=0)
                          

    label=Label(root,font=('Times New Roman',23),text='Entre Username & Password',borderwidth=0,relief="flat",bg="#4FB576")
    label.pack()
    username = "abc" #that's the given username
    password = "cba" #that's the given password

    #username entry
    user_name = Label(root,font=('Calibri Body',11),
                                    text = "Username:",bg="#4FB576").place(x = 65,
                                                                                    y = 42)
    username_entry = Entry(root,width = 35)
    username_entry.pack()


    #password entry
    user_password = Label(root,font=('Calibri Body',11,),
                                            text = "Password:",bg="#4FB576").place(x = 65,
                                                                                            y = 60)
                              
    password_entry = Entry(root, show='*',width=35)
    password_entry.pack()
   

    def trylogin():
        global na
        global count
                #this method is called when the button is pressed
        #to get what's written inside the entries, I use get()
        #check if both username and password in the entries are same of the given ones
      
        df1=pd.read_csv(r'/home/sageiac/Downloads/Bookss.csv')
        if username_entry.get()=="":
            label1.config(text="plz enter your name")
        elif username_entry.get().capitalize() in list(df1.professor):
            
            label1.config(text="name already present")
            password_entry.delete(0,END)
            
        elif password == password_entry.get():
            print("Correct")
            na=username_entry.get()
            if na not in list(df1.professor):
               new={"professor":na.capitalize()}
               print("lll")
               df1=df1.append(new,ignore_index=True)
               df1.to_csv(r'/home/sageiac/Downloads/Bookss.csv',index=False)
               print(df1)
            
                    
        else:
            print("Wrong")
            na=""
            label1.config(text="inncorrect password")
            password_entry.delete(0,END)
            count+=1
            if count==3:
                root.destroy()
            
        return na
    label1=Label(root,font=('helvetica',10,),text="",bg="#4FB576",borderwidth=0)
    label1.pack()
    #when you press this button, trylogin is called
    button = Button(root,font=('Times New Roman',15,'bold'),text="Submit",bg="#4FB576", command = trylogin)
    
    button.pack()
    
    label=Label(root,font=('helvetica',7,),text='Creidentials Login Details!',bg="#4FB576")
    #label.config(bg='GREEN')
    label.pack()
    #App starter
    root.mainloop()
    
    

    
def update():
    global na
    print(na)
    lis=[]
    z=[]
    g=0
    face_cascade = cv2.CascadeClassifier(r'/home/sageiac/Downloads/face_dectect.xml')
    cap = cv2.VideoCapture('/dev/video0')
    while True:
        _, frame = cap.read()
        frame=cv2.flip(frame,1)
        image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        f=cv2.waitKey(1)
        if f%256 == 32 or g!=0:
                g=1
                if len(lis)==0:# this condition is given to excute code only once
                            # making timer of 6 seconds
                            lis.append(1)
                          
                           
                            z=[]
                            n=1
                            print("ll")
                            
                if len(lis)!=0:# for displaying 5 second timer on screen without freezing frame
                     cv2.putText(frame,str(n),(50,50),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),2)
                     k=datetime.now().strftime("%S")[-1]
                     if len(z)==0:
                         z.append(k)
                     if z[0]!=k:
                         
                         z=[]
                         n+=1
                     if n==6:
                         try:
                                         # for face dectection so that you can take croped image of your face
                            for x, y, w, h in face:
                                
                                  face_roi = frame[y-80:y+h+80, x-80:x+w+80]
                                

                            
                            cv2.imwrite(r'/home/sageiac/Desktop/images/' +"Abhishek"+".png",face_roi)
                            break
                           
                             
                            
                         except:
                                
                                lis=[]
                                print("try again")
                                g=0
                                pass
                    
           
        cv2.imshow('cam star', frame)
        try:
           
           if cv2.waitKey(10) == ord('q'):
            
                break
          
        except:
            print("pp")
            pass
    cap.release()
    cv2.destroyAllWindows()




def facee():
    global m
    face_cascade = cv2.CascadeClassifier(r'/home/sageiac/Downloads/face_dectect.xml')
    df=pd.read_csv(r'/home/sageiac/Downloads/Bookss.csv')
    video=cv2.VideoCapture('/dev/video0')
    video.set(cv2.CAP_PROP_BUFFERSIZE,1)
    path=r'/home/sageiac/Desktop/images'
    photo=[r'/home/sageiac/Desktop/images'+"/"+i for i in os.listdir(path)]
    known_face_encod=[face_recognition.face_encodings(face_recognition.load_image_file(i))[0] for i in photo]
    
    known_face_name=[i[:i.index(".")] for i in os.listdir(path)]
    unknown=""
    while True:
        try:
            ret,frame=video.read()
            frame=cv2.flip(frame,1)

            
            frame_copy=frame.copy()
            ha=frame.copy()
            frame=cv2.resize(frame,(0,0),fx=0.20,fy=0.20)
            
            ha=cv2.resize(ha,(0,0),fx=0.50,fy=0.50)
            gray = cv2.cvtColor(ha, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                 
            
            rgb_frame=frame[:,:,::-1]
            
            
            img1_loc=face_recognition.face_locations(rgb_frame)
            img1_encod=face_recognition.face_encodings(rgb_frame,img1_loc)
            
            if len(img1_encod)==0:

               unknown=""
        except: print("error")
        
        for x, y, w, h in face:
                   
                cv2.rectangle(ha,(x,y),(x+w,y+h),(255,255,0),2)
                for (top,right,bottom,left), face_encoding in zip(img1_loc,img1_encod):
                    match=face_recognition.compare_faces(known_face_encod,face_encoding)
                    
                    
                    
                    #emotion dectection
                   
                    name="Unknown"
                    if True in match:
                        index=match.index(True)
                        name=known_face_name[index]
                        
                        if name!="Unknown":
                            try:
                                df.loc[list(df.professor).index(name),datetime.now().strftime("%d-%m-20%y").zfill(10)]="y"
                                lis.append(name)
                            except:pass
                                
                           
                    
                    #updating unknown person's name in attendence sheet
                    if name=="Unknown":
                        
                       
                        
                       print("niu")
                       
                       if cv2.waitKey(10) == ord('u'):
                           upda="yes"                               
                           
                       
                       
                    unknown="1"
                    
                    
                    cv2.putText(ha,"Name: "+name,(0,20),cv2.FONT_HERSHEY_DUPLEX,0.65,(0,0,0),2)
                    
                    cv2.putText(ha,DeepFace.analyze(frame,enforce_detection=False,actions=["emotion"])["dominant_emotion"],(0,40),cv2.FONT_HERSHEY_DUPLEX,0.65,(0,0,0),2)
                    

                    
                    
                    
                    print(name)
                
        cv2.imshow("hh",ha)
        try:
            if cv2.waitKey(10) == ord('q'):
              m="break"  
              break
            if upda=="yes":
                break
        except:
            pass
        
    video.release()
    cv2.destroyAllWindows()
    df.to_csv(r'/home/sageiac/Downloads/Bookss.csv',index=False)
    



while True:
    facee()

    try:
      if m=="break":
        break
    
    except:
         name()
         print(na)
         if na=="":
             
             continue
          
         update()
