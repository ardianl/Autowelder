import math, base64, pickle, PySpin, time, threading, queue
import sys, os, cv2, re, pdb
import PySimpleGUI as sg
import numpy as np
from timeit import default_timer as timer
import ctypes

#initialized_vars = pickle.load(open('pGUI.txt.','rb'))
sg.theme('Reds')
# Define the window layout-----------------------------------------------------------------------------------------------

#Layout for Camera 0~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~
class Instance: #Creates pyGUI window for each camera instance
    def __init__(self, settings):
        #self.cam = None
        self.start = 0
        self.config= {'SQUARE LOW H':0,'SQUARE HIGH H':180,'SQUARE LOW S':0,'SQUARE HIGH S':255,'SQUARE LOW V':0,'SQUARE HIGH V':255,'SQUARE CANNY A':100,'SQUARE CANNY B':200,'SQUARE CANNY C':1,'SQUARE X':0,'SQUARE Y':0,'sq_x1':600,'sq_x2':900,'sq_y1':600,'sq_y2':900,'Area Height':200,'Area Width':200,'Box Height':40,'Box Width':70,'Gap Threshold':100,'Light Threshold':50,'cMin':100,'cMax':1500,'Fixture':False,'f_x1':600,'f_x2':900,'f_y1':600,'f_y2':900,
                      'FixtureBox':True,'FixtureVisible':False,'ROTATE':False,'wRmax':100,'Legacy':True,'Exposure':5195,'Gain':20,'Output':'None','t_min':1,'t_max':255,'CANNY V':31,'morph':1,'morph kernel':1,'dilate':1,'h_lineThresh':50,'v_lineThresh':50,'KERNEL':10,'LOCATION':(10,100),'ix':279,'iy':286,'ox':290,'oy':295,'icirr':82,'ocirr':147,'rx1':262,'rx2':692,'ry1':423,'ry2':793,'region_selected':True,'LOW H':6,'HIGH H':26,'LOW S':0,'HIGH S':255,'LOW V':0,'HIGH V':255,'thresh_val':84,'e_x':770,'e_y':10,'e_w':96,'e_h':59,'w_x':114,'w_y':301,'w_w':472,'w_h':440,'v_x1':709,'v_y1':419,'v_x2':866,'v_y2':671,'h_x1':340,'h_y1':273,'h_x2':663,'h_y2':441,'RESIZE':50.0,'CIRCLES':True,'OUTLINE':False,'ALL CIRCLES':False,'BOXES':True,'EDGES':True,'Weld Status':True,'RCenter':True,'X Shift':50.0,'Y Shift':50.0,'Th':10.0,'0':True,'THRESH':False,'THRESH SLIDER MIN':28.0,'THRESH SLIDER MAX':255.0,'CANNY':False,'CANNY A':7.0,'CANNY B':37.0,'BLUR':False,'BLUR VAL':5.0,'HSV':False,'CANNY LOW H':6.0,'CANNY HIGH H':26.0,'CANNY LOW S':0.0,'CANNY HIGH S':255.0,'CANNY LOW V':0.0,'CANNY HIGH V':255.0,'HOUGH LOW H':6.0,'HOUGH HIGH H':26.0,'HOUGH LOW S':0.0,'HOUGH HIGH S':255.0,'HOUGH LOW V':0.0,'HOUGH HIGH V':255.0,'V VECTOR':213.0,'H VECTOR':316.0,'minR':52.0,'maxR':137.0,'minDist':1,'PARAM1':1,'PARAM2':20.0,'PARAM3':15.0,'lineThresh':0,'minLineLength':50,'maxLineGap':15,}
        
        if settings is not None: self.config.update(settings)

        self.tab0 = [
            [sg.CBox('Use Fixture Point',     size=(35,1), key='Fixture',        default=self.config['Fixture'])],
            [sg.CBox('Reverse X Offset',          size=(35,1), key='REVERSEX',       default=False)],
            [sg.CBox('Reverse Y Offset',          size=(35,1), key='REVERSEY',       default=False)],
            [sg.CBox('Hide Overlay',          size=(35,1), key='HIDE ALL',       default=False)],
            [sg.CBox('Show Fixture Area',     size=(35,1), key='FixtureBox',     default=self.config['FixtureVisible'])],
            [sg.CBox('Show Fixture Features', size=(35,1), key='FixtureVisible', default=self.config['FixtureVisible'])],
            [sg.Frame('Fixture',[
            [sg.Text('X Offset',size=(10,1)),   sg.Slider((0, 500),self.config['SQUARE X'],    1,orientation='h',size=(36, 10),key='SQUARE X')],
            [sg.Text('Y Offset',size=(10,1)),   sg.Slider((0, 500),self.config['SQUARE Y'],    1,orientation='h',size=(36, 10),key='SQUARE Y')],
            [sg.Text('Box Height',size=(10,1)), sg.Slider((10, 200),self.config['Box Height'], 1,orientation='h',size=(36, 10),key='Box Height')],
            [sg.Text('Box Width',size=(10,1)),  sg.Slider((10, 200),self.config['Box Width'],  1,orientation='h',size=(36, 10),key='Box Width')]])]    

        ]
        
        self.tab1 = [      
            [sg.Radio('None',       'Radio', True),
             sg.Radio('Fixture HSV','Radio', key='SQUARE HSV'),
             sg.Radio('Gap HSV',    'Radio', key='GAP HSV'),
             sg.Radio('Misc',       'Radio', key='SQUARE MISC')],
            [sg.Frame('Image',[
            [sg.Text('Blur/Kernel',size=(8,1)),
             sg.Slider((0, 11),self.config['BLUR VAL'],1,orientation='h',size=(18, 10),key='BLUR VAL'),
             sg.Slider((1, 10),self.config['KERNEL'],  1,orientation='h',size=(18, 10),key='KERNEL')]])],
            [sg.Frame('Fixture',[
            [sg.Text('Size',size=(4,1)),
             sg.Slider((0, 100),self.config['cMin'],          1,orientation='h',size=(20, 10),key='cMin'),
             sg.Slider((0, 600),self.config['cMax'],          1,orientation='h',size=(20, 10),key='cMax')],
            [sg.Text('HUE',size=(4,1)),
             sg.Slider((0, 180),self.config['SQUARE LOW H'],  1,orientation='h',size=(20, 10),key='SQUARE LOW H'),
             sg.Slider((0, 180),self.config['SQUARE HIGH H'], 1,orientation='h',size=(20, 10),key='SQUARE HIGH H')],
            [sg.Text('SAT',size=(4,1)),
             sg.Slider((0, 255),self.config['SQUARE LOW S'],  1,orientation='h',size=(20, 10),key='SQUARE LOW S'),
             sg.Slider((0, 255),self.config['SQUARE HIGH S'], 1,orientation='h',size=(20, 10),key='SQUARE HIGH S')],
            [sg.Text('VAL',size=(4,1)),
             sg.Slider((0, 255),self.config['SQUARE LOW V'],  1,orientation='h',size=(20, 10),key='SQUARE LOW V'),
             sg.Slider((0, 255),self.config['SQUARE HIGH V'], 1,orientation='h',size=(20, 10),key='SQUARE HIGH V')]])],
            [sg.Frame('Gap',[
            [sg.Text('Light Threshold', size=(12,1)), sg.Slider((0, 255),  self.config['Light Threshold'], 1,orientation='h',size=(34, 10),key='Light Threshold')],
            [sg.Text('Gap Threshold',   size=(12,1)), sg.Slider((1, 200),  self.config['Gap Threshold'],   1,orientation='h',size=(34, 10),key='Gap Threshold')],
            [sg.Text('Highlight Height',size=(12,1)), sg.Slider((100, 300),self.config['Area Height'],     1,orientation='h',size=(34, 10),key='Area Height')],
            [sg.Text('Highlight Width', size=(12,1)), sg.Slider((100, 300),self.config['Area Width'],      1,orientation='h',size=(34, 10),key='Area Width')]])]
        ]
        
        self.tab2 = [
            [sg.Radio('None',    'Debug', True,  key='none'),
             sg.Radio('Bitwise', 'Debug', False,  key='bitwise'),
             sg.Radio('Overlay', 'Debug', False,  key='Overlay')],
            [sg.CBox('Load Image',        False,  key='imload')],
            [sg.Text('Image Select',size=(9,1)),
             sg.Slider((1, 10),1,                        1,orientation='h', size=(20,15),key='imsave')],
            [sg.Frame('Camera Settings',[
            [sg.Text('Save and exit to apply settings')],
            [sg.Text('FPS',size=(9,1)),     sg.InputText(self.config['FPS'],     size=(9,1),  key='FPS')],
            [sg.Text('Exposure',size=(9,1)),sg.InputText(self.config['Exposure'],size=(9,1),  key='Exposure')],
            [sg.Text('Gain',size=(9,1)),    sg.InputText(self.config['Gain'],    size=(9,1),  key='Gain')]])]
        ]
        
        self.col = [
            [sg.Text('Window Size:'),sg.Slider((1, 100),self.config['RESIZE'],1,orientation='h',size=(36, 15),key='RESIZE')],
            [sg.Button('Select Front Edge'),sg.Button('Select Fixture'),sg.Button('Capture Image'),sg.Button('Save and Exit')],
            [sg.TabGroup([[sg.Tab('Operation',self.tab0), sg.Tab('Settings', self.tab1),sg.Tab('Debugging',self.tab2)]])]            
        ]

        self.layout = [
            [sg.Image(filename="", key='IMG',enable_events=True),sg.Column(self.col)]
        ]
        
        self.window = sg.Window(self.config['CAM ID'], self.layout, location=self.config['LOCATION'])

#### MAIN LOOP #####################################################################################################################################################################
    def loop(self):
        self.cycle = timer() - self.start; self.start = timer();
        self.event, self.values = self.window.read(timeout=20)
        if self.event == sg.WIN_CLOSED           : return False#os._exit(0)
        if self.event == 'Select Front Edge'     : self.image.overlay *= False; cv2.putText(self.image.overlay,'Click and drag to draw box, place center of cross at center of front edge',(150,self.image.height-64),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,255,255),1); self.config.update(dict(zip(['sq_x1','sq_x2','sq_y1','sq_y2'],self.image.select_region(self.window.CurrentLocation()))))
        if self.event == 'Select Fixture'        : self.image.overlay *= False; cv2.putText(self.image.overlay,'Click and drag to draw box around Fixture Square',(150,self.image.height-64),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,255,255),1); self.config.update(dict(zip(['f_x1','f_x2','f_y1','f_y2'],self.image.select_region(self.window.CurrentLocation()))))
        if self.event == 'Save and Exit'         : self.save_and_exit({'LOCATION':self.window.CurrentLocation()})#Saves all settings to file
        if self.event == 'Capture Image'         : pickle.dump(self.cam.get_image(self.config['ROTATE']), open('image{}.txt'.format(str(int(self.values['imsave']))), 'wb'))
                
        if not self.values['imload']:
            self.image = Frame(self.cam.get_image(),self.config['ROTATE'])#Get image from camera and pre-process, creates image object with layers as properties
        else:
            try:    self.image = Frame(pickle.load(open('image{}.txt'.format(str(int(self.values['imsave']))),'rb')),self.config['ROTATE'])
            except: self.image = Frame(self.cam.get_image(),self.config['ROTATE'])
        
        cv2.putText(self.image.overlay, 'Cycle:%.3f'%self.cycle, (30,self.image.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)#display cycle time
        
        try:
            sqx, sqy = self.image.find_gap(self.image.blur(self.image.img,self.values['BLUR VAL'], (int(2*self.values['KERNEL']+1))),
                self.values['Fixture'],self.values['FixtureBox'],self.values['FixtureVisible'],
                self.values['SQUARE LOW H'],self.values['SQUARE LOW S'],self.values['SQUARE LOW V'],
                self.values['SQUARE HIGH H'],self.values['SQUARE HIGH S'],self.values['SQUARE HIGH V'],
                self.config['f_x1'],self.config['f_x2'],self.config['f_y1'],self.config['f_y2'],
                self.config['sq_x1'],self.config['sq_x2'],self.config['sq_y1'],self.config['sq_y2'],
                self.values['SQUARE X'],self.values['SQUARE Y'],self.values['REVERSEX'],self.values['REVERSEY'],
                100*self.values['cMin'],100*self.values['cMax'])
        except:raise(Exception)
        
            
        self.image.highlight(0,0,0,180,255,self.values['Light Threshold'],
                             int(sqx),#+self.values['SQUARE X']),#X GAP CENTER
                             int(sqy),#+self.values['SQUARE Y']),#Y GAP CENTER
                             int(self.values['Box Width']),int(self.values['Box Height']),
                             int(self.values['Area Width']),int(self.values['Area Height']),int(self.values['Gap Threshold']))

        self.config['Output'] = (self.values['SQUARE HSV']*'SQUARE HSV'+self.values['GAP HSV']*'GAP HSV'+self.values['Overlay']*'Overlay'+self.values['bitwise']*'bitwise')
        self.image.overlay *= not self.values['HIDE ALL']
        self.window['IMG'].update(data=self.image.output(self.values['RESIZE'],self.config['Output']))
        return True
####################################################################################################################################################################################

    def save_and_exit(self, location):
        self.config.update(location)
        self.config.update(self.values)
        pickle.dump(self.config, open('{}Psave.txt'.format(self.config['CAM ID']), 'wb'))
        self.cam.exit()
        self.window.close()
        os._exit(0)

    def exit_unsaved(self):
        self.cam.exit()

class Frame:   
    def __init__(self, array, rotation):
        self.img         = array
        self.img         = cv2.cvtColor(self.img,cv2.COLOR_BayerRG2RGB)
        if rotation:     self.img = np.rot90(self.img,3)#
        self.img         = cv2.pyrDown(self.img)
        self.height      = self.img.shape[0]
        self.width       = self.img.shape[1]
        self.empty       = np.zeros_like(self.img)
        self.overlay     = np.zeros_like(self.img)
        self.cannyThresh = np.zeros_like(self.img)
        self.houghThresh = np.zeros_like(self.img)

    def output(self, *args):
        try:
            if args[1]   == 'SQUARE HSV': self.out = self.squareThresh
            elif args[1] == 'GAP HSV'   : self.out = self.gapThresh       
            elif args[1] == 'Overlay'   : self.out = self.overlay
            elif args[1] == 'bitwise'   : self.out = cv2.bitwise_or(self.img, self.overlay)
            else                        : self.out = np.where(self.overlay == 0, self.img, self.overlay)
            self.out = cv2.resize(self.out,(int(self.out.shape[1]*args[0]/100),int(self.out.shape[0]*args[0]/100)))
            self.imgbytes = cv2.imencode('.png',self.out)[1].tobytes()
            return self.imgbytes
        except: raise(Exception)

    def blur(self, array, blur, kernel):
        out = cv2.GaussianBlur(array, (kernel,kernel), blur)
        return out

    def thresh(self, array, h,s,v,H,S,V,*args):
        out = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
        out = cv2.inRange(out, (h,s,v),(H,S,V))
        return out

    def select_region(self, loc):
        #cv2.putText(self.img,'Click and drag to draw box, place center of cross at center of front edge',(150,self.height-64),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,255,255),1)
        cv2.putText(self.img,'SELECT WITH MOUSE, SPACEBAR TO CONFIRM',(50,self.height-24),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(1,255,255),2)
        cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('ROI', loc[0], 0)
        (x,y,w,h) = cv2.selectROI('ROI', cv2.bitwise_or(self.img, self.overlay))#cv2.pyrDown(cv2.bitwise_or(self.img, self.overlay)))
        cv2.destroyAllWindows()
        return [int(x),int(x+w),int(y),int(y+h)]


    def find_gap(self,array,Fixture,FixtureBox,FixtureVisible,h,s,v,H,S,V,fx1,fx2,fy1,fy2,x1,x2,y1,y2,xoff,yoff,reversex,reversey,cmin,cmax):
        if reversex:
            xoff *= -1
        if reversey:
            yoff *= -1
        self.squareThresh = self.thresh(array, h, s, v, H, S, V)
        if Fixture:
            contours, _hierarchy = cv2.findContours(self.squareThresh[fy1:fy2,fx1:fx2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if FixtureBox: cv2.rectangle(self.overlay, (fx1, fy1), (fx2,fy2), (36,255,12), 2)
            for contour in contours:
                area = cv2.contourArea(contour)
                x,y,w,h = cv2.boundingRect(contour)
                if FixtureVisible:
                    cv2.drawContours(self.overlay[fy1:fy2,fx1:fx2], contours, -1, (36,255,12), 2)
                    if area > 10: cv2.putText(self.overlay, str(int(area/100)), (int(fx1+x+w/2),int(fy1+y+h+5)), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)
                if area > cmin and area < cmax:
                    cv2.putText(self.overlay, 'Fixture:({},{}),  {:.0f}'.format(int(fx1+x+w/2),int(fy1+y+h/2),area/100), (600,self.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)
                    cv2.circle(self.overlay, (int(fx1+x+w/2),int(fy1+y+h/2)), 10, (255,36,255), -1)
                    if FixtureVisible:cv2.rectangle(self.overlay[fy1:fy2,fx1:fx2], (x, y), (x + w, y + h), (255,36,12), 2)
                        
                    return (fx1+xoff+x+w/2),(fy1+yoff+y+h/2)
        return (x1+x2)/2, (y1+y2)/2

        

    def highlight(self,h,s,v,H,S,V,x,y,bw,bh,W,A,G):
        self.gapThresh = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.gapThresh = cv2.inRange(self.gapThresh, (h,s,v),(H,S,V))
        #[y-bh:y+bh,x-bw:x+bw]
        try:
            self.gap_area = np.sum(self.gapThresh[y-bh:y+bh,x-bw:x+bw]/255)
            self.highlight = np.zeros_like(self.empty)
            if self.gap_area < G:
                self.highlight[:,:,:] = (0,255,0) #GREEN IMAGE
                cv2.putText(self.overlay, 'OK'.format(self.gap_area), (500,70), cv2.FONT_HERSHEY_DUPLEX, 3, (1,255,1),5)
            else                :
                self.highlight[:,:,:] = (0,0,255) #RED IMAGE
                cv2.putText(self.overlay, 'CHECK', (450,70), cv2.FONT_HERSHEY_DUPLEX, 3, (1,255,255),5)
            self.highlight = cv2.bitwise_and(self.highlight,self.highlight, mask=self.gapThresh)#Cuts threshold out of RED or GREEN image
            self.overlay[y-A:y+A,x-W:x+W] = np.where(self.highlight[y-A:y+A,x-W:x+W] == 0, self.overlay[y-A:y+A,x-W:x+W], self.highlight[y-A:y+A,x-W:x+W])#Applies highlight to area inside ROI
            cv2.putText(self.overlay, 'Gap:{:.0f}'.format(self.gap_area), (300,self.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)
            cv2.rectangle(self.overlay, (x-bw,y-bh), (x+bw,y+bh), (255,1,1),3)
        except:
            raise(Exception)

    
    
class Camera: #INITIALIZE SELECTED CAMERA, SET CAMERA PROPERTIES (STREAM BUFFER, FPS, EXPOSURE, GAIN)
    def __init__(self, cam_id, Gain, Exposure, FPS):
        self.system = PySpin.System.GetInstance()
        self.list = self.system.GetCameras()
        self.cam = self.list.GetBySerial(cam_id)
        #self.cam = PySpin.System.GetInstance().GetCameras().GetBySerial(cam_id)
        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()
        print('Starting camera {}  '.format(cam_id),end='')  
        PySpin.CEnumerationPtr(self.cam.GetTLStreamNodeMap().GetNode('StreamBufferHandlingMode')).SetIntValue(PySpin.CEnumerationPtr(self.cam.GetTLStreamNodeMap().GetNode('StreamBufferHandlingMode')).GetEntryByName('NewestOnly').GetValue())
        PySpin.CEnumerationPtr(self.nodemap.GetNode('ExposureAuto')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('ExposureAuto')).GetEntryByName('Off').GetValue())
        PySpin.CEnumerationPtr(self.nodemap.GetNode('GainAuto')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('GainAuto')).GetEntryByName('Off').GetValue())
        PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode')).GetEntryByName('Continuous').GetValue())
        PySpin.CBooleanPtr(self.nodemap.GetNode('AcquisitionFrameRateEnable')).SetValue(True)
        PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime')).SetValue(Exposure)
        PySpin.CFloatPtr(self.nodemap.GetNode('Gain')).SetValue(Gain)
        PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate')).SetValue(FPS)
        print('\nExposure = {:.2f}\nGain = {:.2f}\nFPS = {:.2f}'.format(PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('Gain')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate')).GetValue()),end='\n')
        self.cam.BeginAcquisition()
    
    def get_image(self):#RETRIEVE LATEST IMAGE FROM CAMERA
        return self.cam.GetNextImage().GetData().reshape(self.height,self.width,1)

    def exit(self, *args):#SHUT DOWN AND EXIT CAMERA
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.list.Clear()
        del self.cam
        self.system.ReleaseInstance()

def get_window():
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    titles = []
    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append(buff.value)
        return True
    EnumWindows(EnumWindowsProc(foreach_window), 0)
    return(titles)

def startup():#Checks for connected cameras and settings file, loads preexisting settings for known cameras and returns settings dictionary
    welders = {'20348005':'PW1','20347994':'PW3','20227035':'PWTest'}
    settings = {'Exposure':5195,'Gain':20,'FPS':10}
    cams = {'cam0':None,'cam1':None}
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    
    #cam_list = PySpin.System.GetInstance().GetCameras()
    open_windows = get_window()
    
    for i in range(cam_list.GetSize()):
         cam_list.GetByIndex(i).Init()
         cams.update({list(cams.keys())[i]:PySpin.CStringPtr(cam_list.GetByIndex(i).GetNodeMap().GetNode('DeviceID')).GetValue()})
         cam_list.GetByIndex(i).DeInit

    #for cam in cam_list:
    #    cam.Init()
    #    cams.update({list(cams.keys())[i]:PySpin.CStringPtr(cam.GetNodeMap().GetNode('DeviceID')).GetValue()})
    #    cam.DeInit()
        
    N_cams = len([x for x in cams.values() if x is not None])
    if N_cams == 0:
        sg.popup('Error no Cameras detected, please try again')
        os._exit(0)
    if N_cams == 1:
        print('Camera detected attempting to load settings... ', end='')
        if os.path.isfile('{}Psave.txt'.format(cams['cam0'])):
            settings = pickle.load(open('{}Psave.txt'.format(cams['cam0']),'rb'))
        else:
            v = cam_setup(settings, list(cams.values())[0], welders[list(cams.values())[0]])
            settings.update(v)
    if N_cams == 2:
        print('Multiple cameras detected...')        
        options = [v for k, v in welders.items() if k in list(cams.values()) and k not in open_windows]
        if len(options) == 0:
            print('Already running both cameras!')
            os.system('pause')
            os._exit(0)
        elif len(options) == 2:
            config_layout = [
                [sg.Text('Choose a camera:',    justification='left'),],
                [sg.Listbox(values=options,size=(15, 3), key='CAM ID'),sg.Text('',size=(2,3))],
                [sg.Submit(), sg.Cancel()]]
            window = sg.Window('Window Title', config_layout)    
            e, v = window.read(close=True)
            if e == sg.WIN_CLOSED: os._exit(0)
            if e == 'Cancel'     : os._exit(0)
            if v['CAM ID'] == [] : v['CAM ID'] = options[0]
            else                 : v['CAM ID'] = v['CAM ID'][0]
            selected_welder = v['CAM ID']
            
        elif len(options) == 1:
            selected_welder = options[0]
            print('One camera already in use, running {}'.format(selected_welder))
            
        selected_cam = list(welders.keys())[list(welders.values()).index(selected_welder)]
        if os.path.isfile('{}Psave.txt'.format(selected_cam)):
            settings = pickle.load(open('{}Psave.txt'.format(selected_cam),'rb'))
        else:
            v = cam_setup(settings, selected_cam, selected_welder)
            settings.update(v)
            
    #settings.update(pickle.load(open('GUIsave.txt','rb')))
    #settings.update({'LOCATION':(10,100)})
    settings.update(cams)
    cam_list.Clear()
    return settings

def cam_setup(settings, cam, welder):
    print('No settings found, starting from scratch...')
    
    config_layout = [
        [sg.Text('Choose settings for {0}:'.format(welder),    justification='left'),],
        [sg.Frame('Camera Settings',[
        [sg.CBox('Rotate Image',key='ROTATE',default=False,)],
        [sg.Text('FPS',size=(9,1)),sg.InputText(settings['FPS'],size=(9,1),key='FPS')],
        [sg.Text('Exposure',size=(9,1)),sg.InputText(settings['Exposure'],size=(9,1),key='Exposure')],
        [sg.Text('Gain',size=(9,1)),sg.InputText(settings['Gain'],size=(9,1),key='Gain')]])],
        [sg.Submit(), sg.Cancel()]]
    e, v = sg.Window('Window Title', config_layout).read(close=True)    
    #e, v = window.read()
    if e == sg.WIN_CLOSED: os._exit(0)
    if e == 'Cancel': os._exit(0)
    v.update({'CAM ID':cam})    
    return v
    
def main():
    loop = True
    settings = startup()
    #print(settings)
    win = Instance(settings)
    win.cam = Camera(settings['CAM ID'],int(settings['Gain']),int(settings['Exposure']),int(settings['FPS']))
        
    while loop:
        loop = win.loop()        

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats()
