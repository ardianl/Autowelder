import math, base64, pickle, PySpin
import sys, os, cv2, re
import PySimpleGUI as sg
import numpy as np
from timeit import default_timer as timer

#initialized_vars = pickle.load(open('pGUI.txt.','rb'))
sg.theme('DarkTeal')
# Define the window layout-----------------------------------------------------------------------------------------------

#Layout for Camera 0~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~
class Instance:
    def __init__(self, settings):
        self.cam = None
        self.start = 0
        self.config= {
            'Output':'None',
            'x':210,
            'y':250,
            'FPS':6,
            'CANNY A':0,
            'CANNY B':0,
            'BLURS':0,
            'LOW H':0,
            'HIGH H1':180,
            'LOW S1':0,
            'HIGH S1':255,
            'LOW V1':0,
            'HIGH V1':22,
            'OLD V1':255,
            'AREA THRESH1':50,
            'W BOX':50,
            'H BOX':50,
            'W SCALAR':50,
            'H SCALAR':50,
            'RESIZE':50,
            'LOCATION':(10,100),
            }
        if settings is not None: self.config.update(settings)
        
        self.tab2 =  [
            [sg.Text('These are the settings used for weld detection, do not touch anything')],        
            [sg.Radio('None',       'Radio', True,                             size=(10, 1))],
            
            #[sg.Radio('Thresh',  'Radio', False,                            size=(10, 1), key='THRESH1'),
            #                            sg.Slider((0, 255),0,1,orientation='h',size=(20, 15),key='THRESH SLIDER MIN1',),
            #                            sg.Slider((0, 255),0,1,orientation='h',size=(20, 15),key='THRESH SLIDER MAX1',)],
            
            [sg.Radio('Edges',      'Radio', False,                            size=(10, 1), key='CANNY1'),
                                        sg.Slider((0, 255),self.config['CANNY A'],1,orientation='h',size=(20, 15),key='CANNY A',),
                                        sg.Slider((0, 255),self.config['CANNY B'],1,orientation='h',size=(20, 15),key='CANNY B',),],
            [sg.Radio('Blur',       'Radio', False,                            size=(10, 1), key='BLUR1'),
                                        sg.Slider((1, 11),self.config['BLURS'],   1,orientation='h',size=(40, 15),key='BLURS',),],
        #HSV Sliders (not used)
            
            [sg.Radio('HSV',        'Radio', False,                                                       size=(3, 1),key='HSV1'),
             sg.Text('HUE',size=(3,1)),
                                        sg.Slider((0, 180),self.config['LOW H'],            1,orientation='h',size=(20, 15),key='LOW H'),
                                        sg.Slider((0, 180),self.config['HIGH H1'],           1,orientation='h',size=(20, 15),key='HIGH H1')],
            [sg.Radio('HW',         'Radio', False,                                                       size=(3, 1), key='HWELD1'),
             sg.Text('SAT',size=(3,1)),
                                        sg.Slider((0, 255),self.config['LOW S1'],            1,orientation='h',size=(20, 15),key='LOW S1'),
                                        sg.Slider((0, 255),self.config['HIGH S1'],           1,orientation='h',size=(20, 15),key='HIGH S1')],
            [sg.Text(' ',size=(6,1)),
             sg.Text('VAL',size=(3,1)),
                                        sg.Slider((0, 255),self.config['LOW V1'],            1,orientation='h',size=(20, 15),key='LOW V1'),
                                        sg.Slider((0, 255),self.config['OLD V1'],            1,orientation='h',size=(20, 15),key='OLD V1')],
        ]
        self.tab1 = [
            [sg.Frame(layout=[
            [sg.CBox('Hide All',        key='HIDE ALL1',      default=False),
             sg.CBox('ZOOM', key='ZOOM', default=False)],
            ],title='Options',          relief=sg.RELIEF_SUNKEN)],
            [sg.Text('Light Threshold'),sg.Slider((0, 100),self.config['HIGH V1'],                          1,orientation='h',size=(20, 15),key='HIGH V1')],
            [sg.Text('Area Threshold'), sg.Slider((1, 200),self.config['AREA THRESH1'],                          1,orientation='h',size=(40, 15),key='AREA THRESH1',),],
            [sg.Text('Highlight Height'),sg.Slider((1, 100),self.config['H SCALAR'],                         1,orientation='h',size=(40, 15),key='H SCALAR',),],
            [sg.Text('Highlight Width'),sg.Slider((1, 100),self.config['W SCALAR'],                          1,orientation='h',size=(40, 15),key='W SCALAR',),],
            [sg.Text('Box Height'),     sg.Slider((1, 100),self.config['H BOX'],                          1,orientation='h',size=(40, 15),key='H BOX',),],
            [sg.Text('Box Width'),      sg.Slider((1, 100),self.config['W BOX'],                          1,orientation='h',size=(40, 15),key='W BOX',),],
        ]
        
        self.col = [
            [sg.Text('Resize Window:'), sg.Slider((1, 100),self.config['RESIZE'],                          1,orientation='h',size=(40, 15),key='RESIZE',),],
            [sg.Button('Draw Region')],
            [sg.Button('Save Settings')],
            [sg.TabGroup([[sg.Tab('Operation', self.tab1), sg.Tab('Settings', self.tab2)]])]
        ]

        self.layout = [
            [sg.Image(filename='', key='IMG',enable_events=True),sg.Column(self.col)]
            
        ]

        self.window = sg.Window(self.config['CAM ID'], self.layout, location=self.config['LOCATION'])


    def loop(self):
        self.cycle = timer() - self.start
        self.start = timer()
        self.event, self.values = self.window.read(timeout=20)
        if self.event == sg.WIN_CLOSED: os._exit(0)

        self.image = Frame(self.cam.get_image())#Get image from camera and pre-process, creates image object with layers as properties
        
        self.image.thresh(self.values['LOW H'],self.values['LOW S1'],self.values['LOW V1'],self.values['HIGH H1'],self.values['HIGH S1'],self.values['HIGH V1'])#create threshold based on sliders
        self.image.highlight(self.config['x'], self.config['y'], self.values['W SCALAR'], self.values['H SCALAR'], self.values['W BOX'], self.values['H BOX'], self.values['AREA THRESH1'])#highlight threshold in image
        
        cv2.putText(self.image.img, '%.3f'%self.cycle, (30,1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,50,30),3)#display cycle time
        #print(self.window.CurrentLocation())

        if self.event == 'Draw Region': self.config.update(self.image.select_region())#
        if self.event == 'Save Settings':
            self.values.update({'LOCATION':self.window.CurrentLocation()})
            self.save_and_exit(self.cam, self.config, self.values, )#Saves all settings to file

        if self.values['HIDE ALL1']: self.image.overlay = self.image.empty
        
        if     self.values['HSV1']: self.config['Output'] = 'HSV'
        #elif  self.values['BLUR1']: settings['Output'] = 'BLUR'
        #elif self.values['CANNY1']: settings['Output'] = 'EDGES'
        else                 : self.config['Output'] = 'None'

        if self.values['ZOOM']: self.config['Output'] = 'ZOOM'
        
        self.window['IMG'].update(data=self.image.output(self.values['RESIZE'],self.config['Output']))

    def save_and_exit(self, cam, settings, values):
        self.config.update(self.values)
        pickle.dump(self.config, open('{}save.txt'.format(self.config['CAM ID']), 'wb'))
        self.cam.exit()
        self.window.close()

        

    def propagate(cam, image):
        None      
        
    
    def output(self):
        return self.layout

    def func():
        None
        

class Camera:
    def __init__(self, cam_id, FPS):
        self.system = PySpin.System.GetInstance()
        self.list = self.system.GetCameras()

        
        '''for i in range(self.list.GetSize()):
            self.list.GetByIndex(i).Init()
            self.cams[i] = PySpin.CStringPtr(self.list.GetByIndex(i).GetNodeMap().GetNode('DeviceID')).GetValue()        
        
        if self.list.GetSize() == 0:
            sg.popup('Error no Cameras detected, please try again')
            os._exit(0)
        elif self.list.GetSize() ==1:
            self.cam = self.list.GetByIndex(0)
        else:
            selection = int(sg.popup_get_text('Choose a camera  {0}:{1}  or {2}:{3}'.format(1,self.cams[0],3,self.cams[1]), 'Camera Selection'))
            if selection == 1:  self.cam = self.list.GetByIndex(0)
            elif selection == 2: self.cam = self.list.GetByIndex(1)
            #elif selection == 3: self.cam = 
            else: print(selection)'''
        
        self.cam = self.list.GetBySerial(cam_id)
        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()
        #self.ID = PySpin.CStringPtr(self.nodemap.GetNode('DeviceID')).GetValue()
        print('Starting camera {}  '.format(cam_id),end='')
        PySpin.CBooleanPtr(self.nodemap.GetNode('AcquisitionFrameRateEnable')).SetValue(True)
        self.FPS = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        #if PySpin.IsWritable(self.FPS): self.FPS.SetValue(FPS)
        print('FPS = %.2f'%self.FPS.GetValue(),end='')
        self.cam.BeginAcquisition()
    
    def get_image(self):
        return self.cam.GetNextImage().GetData().reshape(self.height,self.width,1)

    def exit(self, *args):
        self.list.Clear()
        self.cam.EndAcquisition()
        del self.cam
        self.system.ReleaseInstance()
               

class Frame:
    def __init__(self, array):
        self.img     = array
        self.img     = cv2.cvtColor(self.img,cv2.COLOR_BayerRG2RGB)
        self.img     = cv2.pyrDown(self.img)
        self.height  = self.img.shape[0]
        self.width   = self.img.shape[1]
        self.empty   = np.zeros_like(self.img)
        self.overlay = np.zeros_like(self.img)

    def thresh(self, h,s,v,H,S,V,*args):
        self.thresh = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.thresh = cv2.inRange(self.thresh, (h,s,v),(H,S,V))

    def highlight(self,x,y,W,H,w,h,G):#creates rectangle w*h around point (x,y) and highlights threshold parts of image
        try:
            x1, y1 = int(x - 2*w), int(y - 2*h)
            x2, y2 = int(x + 2*w), int(y + 2*h)
            X1, Y1 = int(x - 4*W), int(y - 4*H)
            X2, Y2 = int(x + 4*W), int(y + 4*H)
            self.X1 = X1
            self.Y1 = Y1
            self.X2 = X2
            self.Y2 = Y2
            self.gap_area = np.sum(self.thresh[y1:y2,x1:x2]/255)
            self.highlight = np.zeros_like(self.img)
            if self.gap_area < G: self.highlight[:,:,:] = (0,255,0) #GREEN IMAGE
            else                : self.highlight[:,:,:] = (0,0,255) #RED IMAGE
            self.highlight = cv2.bitwise_and(self.highlight,self.highlight, mask=self.thresh)#Cuts threshold out of RED or GREEN image
            self.overlay[Y1:Y2,X1:X2] = self.highlight[Y1:Y2,X1:X2] #Applies highlight to area inside ROI

            cv2.putText(self.overlay, 'SUM:{}'.format(self.gap_area), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
            #cv2.putText(self.overlay, 'test', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),1)
            cv2.rectangle(self.overlay, (x1,y1), (x2,y2), (0,255,0),1)
        except:
            raise(Exception)
        

    def select_region(self):
        cv2.putText(self.img, 'Click on Region of Interest, Space to confirm',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,100),3)
        cv2.namedWindow('ROI',cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('ROI', 30, 30)
        #cv2.resizeWindow('ROI', self.width//3, self.height//3)
        #cv2.resize(self.img, (self.width//3, self.height//3))
        x, y, _, _ = cv2.selectROI('ROI', cv2.pyrDown(self.img))
        print('x:{}, y:{}'.format(x*2,y*2))
        cv2.destroyAllWindows()
        return {'x': x*2, 'y': y*2}
        
        
    def find_edges(self):
        None

    def get_blur(self):
        None
        
    
    def output(self, *args):
        if args[1] == 'HSV'     : self.out = self.thresh
        #if args[1] == 'BLUR'    : self.out = self.blur
        #if args[1] == 'EDGES'   : self.out = self.edges
        else                    : self.out = cv2.addWeighted(self.overlay, 1, self.img, 1, 0.0)
        
        if args[1] == 'ZOOM'    :
            self.out = self.out[self.Y1:self.Y2,self.X1:self.X2]
            self.out = cv2.pyrUp(self.out)
        self.out = cv2.resize(self.out,(int(self.out.shape[1]*args[0]/100),int(self.out.shape[0]*args[0]/100)))
        self.imgbytes = cv2.imencode('.png',self.out)[1].tobytes()
        return self.imgbytes

def startup():#Checks for connected cameras and settings file, loads preexisting settings for known cameras and returns settings dictionary
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cams = {'cam0':None,'cam1':None}
    names = ['cam0','cam1']
    for i in range(cam_list.GetSize()):
         cam_list.GetByIndex(i).Init()
         cams.update({names[i]:PySpin.CStringPtr(cam_list.GetByIndex(i).GetNodeMap().GetNode('DeviceID')).GetValue()}) 
    N_cams = len([x for x in cams.values() if x is not None])
    
    if N_cams == 0:
        sg.popup('Error no Cameras detected, please try again')
        os._exit(0)
    if N_cams == 1:
        print('Camera detected attempting to load settings... ', end='')
        if os.path.isfile('{}save.txt'.format(cams['cam0'])): settings = pickle.load(open('{}save.txt'.format(cams['cam0']),'rb'))
    if N_cams == 2 or not os.path.isfile('{}save.txt'.format(cams['cam0'])):
        print('No settings found, starting from scratch...')
        settings = {
            'FPS':6
            }
        options = list(cams.values())
        if N_cams == 2: options.append('Both')
        config_layout = [
            [sg.Text('Setup')],
            [sg.Text('Choose a camera:          ',    justification='left'),
             sg.Text('FPS:')],
            [sg.Listbox(values=options,size=(15, 3), key='CAM ID'),sg.Text('',size=(2,3)),#*(len(cams)==2)              
             sg.Spin([i for i in range(35)], settings['FPS'], key='FPS', size=(10,10))],
            [sg.Submit(), sg.Cancel()]
        ]
        window = sg.Window('Window Title', config_layout)    
        _, v = window.read()
        if v['CAM ID'] == []:
            if N_cams == 1: v['CAM ID'] = cams['cam0']
            if N_cams == 2: v['CAM ID'] = 'Both'
        else:
            v['CAM ID'] = v['CAM ID'][0]
        settings.update(v)
        window.close()

    settings.update(cams)
    cam_list.Clear()
    return settings

def load_settings(cam_id):
    if os.path.isfile('{}save.txt'.format(cam_id)): return pickle.load(open('{}save.txt'.format(cam_id),'rb'))
    else                                          : return {'CAM ID':cam_id}
    
    
def main():
    #window.CurrentLocation()
    start = 0
    values = {}
    settings = startup()
    if settings['CAM ID'] == 'Both':
        win0 = Instance(load_settings(settings['cam0']))
        win1 = Instance(load_settings(settings['cam1']))
        wins = [win0, win1]                        
        for win in wins:
            win.cam = Camera(win.config['CAM ID'],settings['FPS'])
    else:
        win = Instance(settings)
        wins = [win]
        win.cam = Camera(settings['CAM ID'], settings['FPS'])
    while True:
        for win in wins:
            win.loop()

if __name__ == '__main__':
    main()
