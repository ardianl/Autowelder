import PySimpleGUI as sg
import math
import base64
import pickle
import PySpin
from simple_pyspin import Camera
import math
import sys
import os
import time
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from collections import deque

initialized_vars = pickle.load(open('GUIsave.txt.','rb'))
print(initialized_vars)
globals().update(initialized_vars)

low_H = 6;         high_H = 26 #
low_S = 0;         high_S = 255 # PRESET VALUES WHICH ARE KNOWN TO PRODUCE GOOD RESULTS
#low_V = 0;         high_V = 255 #


file_name = str(int(time.time()))
imwrite_result = 'NULL'
increment = 0
outcrement = 0
#Initliazie empty lists and variables
weld_list,corner_list,d_y_list,d_x_list = [],[],[],[]
h_line,v_line,w_list=[],[],[]
weld_limit = 10
num=10

def INT(a,b):#Used to turn Tuples into integers
    return [int(x) for x in (a,b)]

def normal_vector(line):#Returns the vector of a line and its normal vector
    vector = (line[1][0]-line[0][0],line[1][1]-line[0][1])
    normal = (-vector[1],-vector[0])
    mag = math.sqrt(vector[0]**2 + vector[1]**2)
    unit_vector = (vector[0]/mag,vector[1]/mag)
    unit_normal = (-unit_vector[1],-unit_vector[0])
    return unit_vector, unit_normal

def line_intersection(line1, line2):#Returns the intersection of two lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
       return 0, 0
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


with Camera() as cam:
    cam = Camera()
    cam.PixelFormat = "BayerRG8"
    cam.AcquisitionFrameRateEnable = True
    cam.AcquisitionFrameRate = 10
    cam.GainAuto = 'Continuous'
    cam.ExposureAuto = 'Continuous'
    #cam.Gamma = 0.8
    cam.GammaEnable = False
    cam.init()
    cam.start()
    #sg.theme("Dark Blue 3")
#while True:
    sg.theme("DarkTeal")
# Define the window layout-----------------------------------------------------------------------------------------------    
    tab2_layout =  [
        [sg.Text('These are the settings used for weld detection, do not touch anything')],        
        [sg.Radio("None", "Radio", True, size=(10, 1))],
        #[sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),sg.Slider((0, 255),globals()["-THRESH SLIDER-"],1,orientation="h",size=(40, 15),key="-THRESH SLIDER-",),],
        [sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),sg.Slider((0, 255),globals()["-THRESH SLIDER MIN-"],1,orientation="h",size=(20, 15),key="-THRESH SLIDER MIN-",),sg.Slider((0, 255),globals()["-THRESH SLIDER MAX-"],1,orientation="h",size=(20, 15),key="-THRESH SLIDER MAX-",)],
        [sg.Radio("canny", "Radio", size=(10, 1), key="-CANNY-"),
         sg.Slider((0, 255),globals()["-CANNY SLIDER A-"],1,orientation="h",size=(20, 15),key="-CANNY SLIDER A-",),
         sg.Slider((0, 255),globals()["-CANNY SLIDER B-"],1,orientation="h",size=(20, 15),key="-CANNY SLIDER B-",),],
        [sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),sg.Slider((1, 11),globals()["-BLUR SLIDER-"],1,orientation="h",size=(40, 15),key="-BLUR SLIDER-",),],
    #HSV Sliders (not used)
        [sg.Radio("HSV", "Radio", size=(3, 1), key="-HSV-"),        
         sg.Text('HUE',size=(3,1)),                        sg.Slider((0, 180),low_H,1,orientation="h",size=(20, 15),key="-LOW H-"),sg.Slider((0, 180),high_H,1,orientation="h",size=(20, 15),key="-HIGH H-")],
        [sg.Radio("HW", "Radio", size=(3, 1), key="-HWELD-"),sg.Text('SAT',size=(3,1)),sg.Slider((0, 255),low_S,1,orientation="h",size=(20, 15),key="-LOW S-"),sg.Slider((0, 255),high_S,1,orientation="h",size=(20, 15),key="-HIGH S-")],
        [sg.Text(' ',size=(6,1)),sg.Text('VAL',size=(3,1)),sg.Slider((0, 255),low_V,1,orientation="h",size=(20, 15),key="-LOW V-"),sg.Slider((0, 255),high_V,1,orientation="h",size=(20, 15),key="-HIGH V-")],
        #[sg.Text('minLineLength'),sg.Slider((1, 100),50,1,orientation="h",size=(40, 15),key="-minLineLength-",),],
        #[sg.Text('maxLineGap'),sg.Slider((1, 50),15,1,orientation="h",size=(40, 15),key="-maxLineGap-",),],
        [sg.Text('Weld Position'),sg.Slider((0, 500),globals()["-V VECTOR-"],1,orientation="h",size=(20, 15),key="-V VECTOR-",),sg.Slider((0, 500),globals()["-H VECTOR-"],1,orientation="h",size=(20, 15),key="-H VECTOR-")],
        [sg.Text('Weld Radius'),sg.Slider((10, 100),globals()["-WELD R MIN-"],1,orientation="h",size=(20, 15),key="-WELD R MIN-",),sg.Slider((50, 200),globals()["-WELD R MAX-"],1,orientation="h",size=(20, 15),key="-WELD R MAX-")],
        #[sg.Text('param1'),sg.Slider((1, 50),1,1,orientation="h",size=(40, 15),key="-PARAM1-",)],
        [sg.Text('param2'),sg.Slider((1, 100),globals()["-PARAM2-"],1,orientation="h",size=(40, 15),key="-PARAM2-")],
    ]
    tab1_layout = [
        [sg.Frame(layout=[
        [sg.CBox('Show Weld Center Guess',    key="-CIRCLES-",      default=True,)],
        [sg.CBox('Show Outter Weld Bound',    key="-OUTLINE-",      default=False,)],
        [sg.CBox('Show Found Circles',key="-ALL CIRCLES-",default=False)],
        [sg.CBox('Show Boxes',      key="-SHOW BOXES-", default=True)],
        [sg.CBox('Show Edges',         key="-SHOW MORE-",  default=True)],
        [sg.CBox('Show Weld Judgement',         key="-Weld Status-",  default=True)],
        [sg.CBox('Show Weld Reset Center',         key="-RCenter-",  default=True)],
        ], title='Options', relief=sg.RELIEF_SUNKEN)],
        [sg.Text('X-Shift'),sg.Slider((1, 100),50,1,orientation="h",size=(40, 15),key="-X Shift-",),],
        [sg.Text('Y-Shift'),sg.Slider((1, 100),50,1,orientation="h",size=(40, 15),key="-Y Shift-",),],
        [sg.Text('Th'),sg.Slider((1, 300),10,1,orientation="h",size=(40, 15),key="-Th-",),],
        #[sg.Text('width'),sg.Slider((1, 300),215,1,orientation="h",size=(40, 15),key="-width-",),],
        #[sg.Text('height'),sg.Slider((1, 300),200,1,orientation="h",size=(40, 15),key="-height-",),],
        #[sg.Text('Vertical Offset'),sg.Slider((1, 300),213,1,orientation="h",size=(40, 15),key="-v_offset-",),],
        #[sg.Text('Horizontal Offset'),sg.Slider((1, 300),286,1,orientation="h",size=(40, 15),key="-h_offset-",),],
        #[sg.Text('Proportion'),sg.Slider((1, 100),25,1,orientation="h",size=(40, 15),key="-porp-",),],
        #[sg.Text('V/N'),sg.Slider((100, 500),200,1,orientation="h",size=(20, 15),key="-V VICTOR-",),sg.Slider((100, 500),200,1,orientation="h",size=(20, 15),key="-V NORMAL-")],
    ]    
    col = [
        [sg.Text('Resize Window:'),sg.Slider((1, 100),80,1,orientation="h",size=(40, 15),key="-RESIZE SLIDER-",),],
        [sg.Button('Select Weld'),sg.Button('Select Horizontal Edge'),sg.Button('Select Vertical Edge')],
        [sg.Button('Save and Exit'),sg.Button('Exit without saving')],
        [sg.Button('Capture Image'),sg.Button('Plus'),sg.Button('Reset Weld')],
        [sg.TabGroup([[sg.Tab('Operation', tab1_layout), sg.Tab('Settings', tab2_layout)]])]
    ]
    layout = [
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-",enable_events=True),sg.Column(col)]
        
    ] 
    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(100, 100))
#-------------------------------------------------------------------------------------------------------------------------
    
#Start of Loop Here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while True:    
        event, values = window.read(timeout=20)
        values["-minLineLength-"],values["-maxLineGap-"] = 50,15#Explicitly defined in case needed later
        if event == "Exit without saving" or sg.WIN_CLOSED:#Shutdown program if window is closed or exit without saving
            cam.stop()
            os._exit(0)
        if event == "Save and Exit":#Saves all variables to save file and exits program
            initialized_vars = {'ix':ix,'iy':iy,'ox':ox,'oy':oy,'icirr':icirr,'ocirr':ocirr,'rx1':rx1,'rx2':rx2,'ry1':ry1,'ry2':ry2,'region_selected':region_selected,'Inner_Circle_placed':Inner_Circle_placed,'Outer_Circle_placed':Outer_Circle_placed,'low_H':low_H,'high_H':high_H,'low_S':low_S,'high_S':high_S,'low_V':low_V,'high_V':high_V,'thresh_val':thresh_val,'e_x':e_x,'e_y':e_y,'e_w':e_w,'e_h':e_h,'w_x':w_x,'w_y':w_y,'w_w':w_w,'w_h':w_h,'v_x1':v_x1,'v_y1':v_y1,'v_x2':v_x2,'v_y2':v_y2,'h_x1':h_x1,'h_y1':h_y1,'h_x2':h_x2,'h_y2':h_y2}
            initialized_vars.update(values)
            pickle.dump(initialized_vars, open('GUIsave.txt', 'wb'))
            cam.stop()
            os._exit(0)
        if increment < 101:  increment += 1
        if outcrement < 101:  outcrement += 1
           
        frame_raw = cam.get_array()
        #frame_raw = np.load('frame_raw.npy')
        frame_color = cv2.cvtColor(frame_raw,cv2.COLOR_BayerRG2RGB)
        frame_down = cv2.pyrDown(frame_color)
        frame_blur = cv2.GaussianBlur(frame_down, (21,21), values["-BLUR SLIDER-"])
        
        frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (values["-LOW H-"], values["-LOW S-"], values["-LOW V-"]), (values["-HIGH H-"], values["-HIGH S-"], values["-HIGH V-"]))
        
        #frame_HLS = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HLS)
        #frame_threshold = cv2.inRange(frame_HLS, (values["-LOW H-"], values["-LOW S-"], values["-LOW V-"]), (values["-HIGH H-"], values["-HIGH S-"], values["-HIGH V-"]))
        #HLS (41,0,0),(180,64,255) circle
        #HLS (0,226,107),(49,255,255) center
        
        edges = cv2.Canny(frame_blur, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"])
        
        frame_thresh = cv2.Canny(frame_threshold, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"])
        frame = frame_down
        
        
        
#Try to find circles approximating weld----------------------------------------------------------------------------------------------------------------------
        try:
            circles = cv2.HoughCircles(frame_thresh[ry1:ry2,rx1:rx2],cv2.HOUGH_GRADIENT,1,1,param1=1,param2=values["-PARAM2-"],minRadius=100,maxRadius=200)
            if circles is not None:
                if values["-ALL CIRCLES-"]:
                        for i in circles[0,:]: cv2.circle(frame_down[ry1:ry2,rx1:rx2,:],(int(i[0]),int(i[1])),int(i[2]),(255,255,255),5)              
                weld = np.around(np.average(circles, axis=1)).flatten().astype(int)
                if len(weld_list) > 15: weld_list.pop()
                weld_list.insert(0,weld)
                if weld_list:
                    weld_centroid = np.average(weld_list,axis=0).astype(int)
            if values["-CIRCLES-"]:
                cv2.circle(frame_down[ry1:ry2,rx1:rx2,:],(weld_centroid[0],weld_centroid[1]),weld_centroid[2],(100,255,100),2)
                cv2.circle(frame_down[ry1:ry2,rx1:rx2,:],(weld_centroid[0],weld_centroid[1]),5,(100,255,100),-1)
        except:
            None
#Skip if none are found--------------------------------------------------------------------------------------------------------------------------------------
            
#Find lines approximating electrode edges____________________________________________________________________________________________________________________
        v_lines = cv2.HoughLinesP(edges[v_y1:v_y2,v_x1:v_x2], 1, np.pi/180, 50, None, values["-minLineLength-"], values["-maxLineGap-"])#Vertical Edge
        h_lines = cv2.HoughLinesP(edges[h_y1:h_y2,h_x1:h_x2], 1, np.pi/180, 50, None, values["-minLineLength-"], values["-maxLineGap-"])#Horizontal Edge
        #a_lines = cv2.HoughLinesP(edges,1,np.pi/180,int(values["-Th-"]),None,values["-minLineLength-"], values["-maxLineGap-"])

        '''if a_lines is not None:
            for line in a_lines:
                line = line[0]
                cv2.line(frame_down,(line[0],line[1]), (line[2],line[3]), (255,0,0), 3, cv2.LINE_AA)'''
        
        if h_lines is not None:#Finds horizontal lines
            h2 = [l[0][1] for l in h_lines]#Extract Y values of each line
            l= h_lines[h2.index(min(h2))][0]#line with Y closest to weld
            P1,P2 = np.asarray((l[0]+h_x1,l[1]+h_y1)),np.asarray((l[2]+h_x1,l[3]+h_y1))#P1 P2 on edge line
            h_line = (P1,P2)
            if values["-SHOW MORE-"]:#Shows line
                cv2.line(frame[h_y1:h_y2,h_x1:h_x2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)#Draw Edge Line
                    
        if v_lines is not None:#Finds vertical lines
            v2 = [l[0][0] for l in v_lines]#Extract X values of each line
            l = v_lines[v2.index(min(v2))][0]#d_x_avg first line (most confident)            
            P1,P2 = np.asarray((l[0]+v_x1,l[1]+v_y1)),np.asarray((l[2]+v_x1,l[3]+v_y1)),#P1 P2 on edge
            v_line = (P1,P2)
            if values["-SHOW MORE-"]:#Shows lines
                cv2.line(frame[v_y1:v_y2,v_x1:v_x2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)#Draw line
           
        if h_line and v_line:#Finds the electrode corner and weld true position from edge lines
            v_vector, v_normal = normal_vector(v_line)#Getting normal vector
            h_vector, h_normal = normal_vector(h_line)
            corner_x,corner_y = line_intersection(h_line,v_line)
            corner = (int(corner_x),int(corner_y))
            if len(corner_list) > 15: corner_list.pop()
            corner_list.insert(0,corner)
            corner_centroid = np.average(corner_list,axis=0).astype(int)
            cv2.circle(frame_down, (corner_centroid[0],corner_centroid[1]), 5, (255,255,0), -1)
            w_center = corner_centroid + tuple([x * values["-V VECTOR-"] for x in v_vector]) - tuple([x * values["-H VECTOR-"] for x in h_vector])
            w_list.insert(0,w_center)
            if len(w_list)>15: w_list.pop()
            w_center = np.average(w_list,axis=0).astype(int)
            cv2.circle(frame_down, (int(w_center[0]),int(w_center[1])), int(values["-WELD R MIN-"]), (255,0,255), 2) 
            if values["-OUTLINE-"]: cv2.circle(frame_down, (int(w_center[0]),int(w_center[1])), int(values["-WELD R MAX-"]), (255,0,255), 2)
            
            if values["-SHOW MORE-"]:
                cv2.putText(frame_down,'Corner: ('+str(corner_centroid[0])+','+str(corner_centroid[1])+')',(5,990),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                cv2.putText(frame_down,'Weld Bound:('+str(int(w_center[0]))+','+str(int(w_center[1]))+')',(5,910),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                cv2.putText(frame_down,'Weld Radius:['+str(int(values["-WELD R MIN-"]))+','+str(int(values["-WELD R MAX-"]))+']',(5,950),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
            
#Judges weld good or bad--------------------------------------------------------------------------------------------------------
            if values["-Weld Status-"]:            
                try:
                    delta = math.sqrt((w_center[0]-weld_centroid[0]-rx1)**2 + (w_center[1]-weld_centroid[1]-ry1)**2)
                    if delta < values["-WELD R MIN-"]:
                        cv2.putText(frame_down,'Weld is GOOD',(5,90),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,0),2)
                    else:
                        cv2.putText(frame_down,'Weld is BAD',(5,90),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),2)
                except:
                    None
#______________________________________________________________________________________________________________________________
        '''try:
            cv2.putText(frame_down,'Weld_X: '+str(d_x_avg),(5,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame_down,'Weld_Y: '+str(d_y_avg),(5,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            if values["-WELD X MIN-"] < d_x_avg < values["-WELD X MAX-"] and values["-WELD Y MIN-"] < d_y_avg < values["-WELD Y MAX-"]:
                cv2.putText(frame_down,'WELD IS GOOD',(5,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame_down,'WELD IS BAD',(5,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        except:
            None'''
        
        
        '''try:
            M = cv2.moments(frame_thresh[e_y:e_y+e_h,e_x:e_x+e_w])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if values["-SHOW MORE-"]:
                cv2.circle(frame_down, (cX+e_x, cY+e_y), 5, (255,0,0), -1)
        except:
            None'''

        

                

        if values["-SHOW BOXES-"] or increment<15:
            try:
                cv2.rectangle(frame_down,(rx1,ry1),(rx2,ry2),        (100,255,100),2)#Weld Box
                cv2.rectangle(frame_down,(v_x1,v_y1),(v_x2,v_y2),(255,100,100),2)#Vertical Edge Box
                cv2.rectangle(frame_down,(h_x1,h_y1),(h_x2,h_y2),(255,100,100),2)#Horizontal Edge Box
                cv2.putText(frame_down,'Weld',(rx1+1,ry2-5),                  cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)#Weld Label
                cv2.putText(frame_down,'Vertical Edge',(v_x1+1,v_y2-5),  cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100),2)#V Edge Label
                cv2.putText(frame_down,'Horizontal Edge',(h_x1+1,h_y2-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100),2)#H Edge Label
                #cv2.rectangle(frame_down,(e_x,e_y),(e_x+e_w,e_y+e_h),(255,100,100),2)
            except:
                raise(Exception)#None

        
        #if increment < 30:            
            #cv2.putText(frame_down,'Reading unstable..',(5,60),cv2.FONT_HERSHEY_SIMPLEX,1,(60,60,255),2)
        if outcrement < 30:
            cv2.putText(frame_down,str(file_name+imwrite_result),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)#file_name+imwrite_result
        
        if event == 'Reset Weld':
            increment = 0
            cv2.putText(frame_down,'Click on the Weld Center, press spacebar to confirm',(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)
            (r_x,r_y,r_w,r_h) = cv2.selectROI(frame_down)
            wX = r_x + r_w//2
            wY = r_y + r_h//2
            cv2.destroyAllWindows()
            try:
                values["-v_offset-"]=213
                values["-h_offset-"]=286
                values["-porp-"]=25
                values["-width-"]=215
                values["-height-"]=185 
                wbox1 = INT(wX-values["-width-"],wY-values["-height-"]+25)
                wbox2 = INT(wX+values["-width-"],wY+values["-height-"]+25)
                vbox1 = INT(wX+values["-h_offset-"]-values["-porp-"]*values["-width-"]/100,wY-0.75*values["-height-"]-25)
                vbox2 = INT(wX+values["-h_offset-"]+values["-porp-"]*values["-width-"]/100+50,wY+0.75*values["-height-"]-50)
                hbox1 = INT(wX-0.75*values["-width-"]+25,wY-values["-v_offset-"]-values["-porp-"]*values["-height-"]/100-50)
                hbox2 = INT(wX+0.75*values["-width-"]+25,wY-values["-v_offset-"]+values["-porp-"]*values["-height-"]/100+25)
                rx1,rx2,ry1,ry2     = wbox1[0],wbox2[0],wbox1[1],wbox2[1]
                v_x1,v_x2,v_y1,v_y2 = vbox1[0],vbox2[0],vbox1[1],vbox2[1]
                h_x1,h_x2,h_y1,h_y2 = hbox1[0],hbox2[0],hbox1[1],hbox2[1]
            except:
                None
        if event == 'Capture Image':
            file_name = str(int(time.time()))
            try:
                cv2.imwrite('C:\\WELDER\\'+file_name+'.png',frame_raw)
                imwrite_result = ' SAVED'
            except:
                imwrite_result = ' ERROR'
                raise(Exception)
            outcrement = 0
        if event == 'Select Weld':
            cv2.putText(frame_down,'Select region with mouse, press spacebar to confirm',(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)
            (w_x,w_y,w_w,w_h) = cv2.selectROI(frame_down)
            rx1,rx2,ry1,ry2 = int(w_x),int(w_x+w_w),int(w_y),int(w_y+w_h)
            increment = 0
            cv2.destroyAllWindows()
        if event == 'Select Vertical Edge':
            cv2.putText(frame_down,'Select region with mouse, press spacebar to confirm',(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)
            (v_x,v_y,v_w,v_h) = cv2.selectROI(frame_down)
            v_x1,v_x2,v_y1,v_y2 = v_x,v_x+v_w,v_y,v_y+v_h
            increment = 0
            cv2.destroyAllWindows()
        if event == 'Select Horizontal Edge':
            cv2.putText(frame_down,'Select region with mouse, press spacebar to confirm',(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,100),2)
            (h_x,h_y,h_w,h_h) = cv2.selectROI(frame_down)
            h_x1,h_x2,h_y1,h_y2 = h_x,h_x+h_w,h_y,h_y+h_h
            increment = 0
            cv2.destroyAllWindows()
        if event == "Plus":
            np.save('frame_raw.npy',frame_raw)
            np.save('frame_color.npy',frame_color)
            np.save('frame_down.npy',frame_down)
            print('Arrays saved')
        if values["-THRESH-"]:
            frame = frame_thresh
        elif values["-CANNY-"]:
            frame = edges
        elif values["-BLUR-"]:
            frame = frame_blur
        elif values["-HSV-"]:           
            frame = frame_threshold
        elif values["-HWELD-"]:           
            frame = frame_threshold

        frame = cv2.resize(frame,(int(frame.shape[1]*values["-RESIZE SLIDER-"]/100),int(frame.shape[0]*values["-RESIZE SLIDER-"]/100)))
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
    window.close()
