#AUTOWELDER PROGRAM
#USED TO MEASURE GAP BETWEEN ELECTRODE AND WELDING FIXTURE, DISPLAYS PASS/FAIL POSITIONING TO OPERATOR (PRESENCE OF GAP LEADS TO WELDS OUTSIDE OF TOLERANCE)
#GAP REGION CAN BE SELECTED AND DETECTION __ARAMETERS FINE TUNED, FIXTURE POINT CAN BE USED TO TRACK MARKED POINT ON WELD FIXTURE KEEPING GAP CHECK BOX ALIGNED WITH ELECTRODE EDGE

import math, base64, pickle, PySpin, time, threading, queue
import sys, os, cv2, re, pdb
import functools
#from functools mport lru_cache
import PySimpleGUI as sg
import numpy as np
from timeit import default_timer as timer
import ctypes
import cProfile, pstats

sg.theme('Reds')

class WelderInstance: #INSTANCE OF CAMERA
    def __init__(self, settings):
        self.start = 0
        self.config= {'Fixture Enable':False, 'Fixture Features Visible':False, 'Fixture Region Visible':False, 'Fixture Threshold':(0,0,0,180,255,255), 'Fixture Offset':(50,50), 'Fixture Reverse':(0,0),
                      'Fixture Region':(0,0,1,1), 'Contour Bounds':(0,0,1,1), 'Highlight Size':(200,200), 'Gap Box Size':(70,40), 'Gap Size Limit':100, 'Gap Light Threshold':50, 'Contour Size':(100,200),
                      'Output':'None', 'Kernel':10, 'Blur Value':5, 'Window Size':50, 'Window Location':(10,100), 'Electrode ROI':(600,900)}
        
        if settings is not None: self.config.update(settings)

        #DEFAULT OPERATOR TAB
        self.tab0 = [
            [sg.CBox('Use Fixture Point',     size=(35,1),   key='Fixture Enable',           default=self.config['Fixture Enable'])],
            [sg.CBox('Show Fixture Region',   size=(35,1),   key='Fixture Region Visible',   default=self.config['Fixture Region Visible'])],
            [sg.CBox('Show Fixture Features', size=(35,1),   key='Fixture Features Visible', default=self.config['Fixture Features Visible'])],
            [sg.CBox('Reverse X Offset',      size=(35,1),   key='Fixture Reverse X',        default=self.config['Fixture Reverse'][0])],
            [sg.CBox('Reverse Y Offset',      size=(35,1),   key='Fixture Reverse Y',        default=self.config['Fixture Reverse'][1])],
            [sg.CBox('Hide Overlay',          size=(35,1),   key='Hide Overlay',             default=False)],
            [sg.Frame('Fixture',[
            [sg.Text('X Offset',              size=(10,1)),
             sg.Slider(range=(0, 600),        size=(36, 10), key='X Offset',                 default_value=self.config['Fixture Offset'][0], orientation='h',)],
            [sg.Text('Y Offset',              size=(10,1)),
             sg.Slider(range=(0, 600),        size=(36, 10), key='Y Offset',                 default_value=self.config['Fixture Offset'][1], orientation='h',)],
            [sg.Text('Box Height',            size=(10,1)),
             sg.Slider(range=(10, 200),       size=(36, 10), key='Gap Box Height',           default_value=self.config['Gap Box Size'][1],    orientation='h',)],
            [sg.Text('Box Width',             size=(10,1)),
             sg.Slider(range=(10, 200),       size=(36, 10), key='Gap Box Width',            default_value=self.config['Gap Box Size'][0],     orientation='h',)]])]
        ]
        
        #SETTINGS TAB, ADJUSTS FIXTURE AND GAP DETECTION SETTINGS
        self.tab1 = [      
            [sg.Radio('None',                 group_id=0,    key='None',                     default=True),
             sg.Radio('Fixture HSV',          group_id=0,    key='Fixture HSV',              default=False),
             sg.Radio('Gap HSV',              group_id=0,    key='GAP HSV',                  default=False),
             sg.Radio('Misc',                 group_id=0,    key='SQUARE MISC',              default=False)],
            [sg.Frame('Image',[
            [sg.Text('Blur/Kernel',           size=(8,1)),
             sg.Slider(range=(0, 11),         size=(18, 10), key='Blur Value',               default_value=self.config['Blur Value'],            orientation='h',),
             sg.Slider(range=(1, 10),         size=(18, 10), key='Kernel',                   default_value=self.config['Kernel'],                orientation='h',)]])],
            [sg.Frame('Fixture',[
            [sg.Text('Size',size=(4,1)),
             sg.Slider(range=(0, 100),        size=(20,10),  key='Contour Min',              default_value=self.config['Contour Size'][0],                  orientation='h'),
             sg.Slider(range=(0, 255),        size=(20,10),  key='Contour Max',              default_value=self.config['Contour Size'][1],                  orientation='h')],
            [sg.Text('HUE',size=(4,1)),
             sg.Slider(range=(0, 180),        size=(20,10),  key='Fixture h',                default_value=self.config['Fixture Threshold'][0],  orientation='h'),
             sg.Slider(range=(0, 180),        size=(20,10),  key='Fixture H',                default_value=self.config['Fixture Threshold'][3],  orientation='h')],
            [sg.Text('SAT',size=(4,1)),
             sg.Slider(range=(0, 255),        size=(20,10),  key='Fixture s',                default_value=self.config['Fixture Threshold'][1],  orientation='h'),
             sg.Slider(range=(0, 255),        size=(20,10),  key='Fixture S',                default_value=self.config['Fixture Threshold'][4],  orientation='h')],
            [sg.Text('VAL',size=(4,1)),
             sg.Slider(range=(0, 255),        size=(20,10),  key='Fixture v',                default_value=self.config['Fixture Threshold'][2],  orientation='h'),
             sg.Slider(range=(0, 255),        size=(20,10),  key='Fixture V',                default_value=self.config['Fixture Threshold'][5],  orientation='h')]])],
            [sg.Frame('Gap',[
            [sg.Text('Light Threshold',       size=(12,1)),
             sg.Slider(range=(0, 255),        size=(34,10),  key='Gap Light Threshold',      default_value=self.config['Gap Light Threshold'],       orientation='h')],
            [sg.Text('Gap Threshold',         size=(12,1)),
             sg.Slider(range=(1, 200),        size=(34,10),  key='Gap Size Limit',           default_value=self.config['Gap Size Limit'],         orientation='h')],
            [sg.Text('Highlight Height',      size=(12,1)),
             sg.Slider(range=(100, 300),      size=(34,10),  key='Highlight Height',         default_value=self.config['Highlight Size'][1],     orientation='h')],
            [sg.Text('Highlight Width',       size=(12,1)),
             sg.Slider(range=(100, 300),      size=(34,10),  key='Highlight Width',          default_value=self.config['Highlight Size'][0],     orientation='h')]])]
        ]

        list_data     = [[''],['No Fixture', 'Check if Fixture enabled'],['Not opening', 'close all programs and restart, try to check and see if the window is appearing on another monitor']]
        headings_data = ['Problem', 'Solution']

        self.tab2 = [
            # [sg.Table(values=data[1:][:], headings=headings, max_col_width=25,
            #         # background_color='light blue',
            #         auto_size_columns=True,
            #         display_row_numbers=True,
            #         justification='right',
            #         num_rows=20,
            #         alternating_row_color='lightyellow',
            #         key='-TABLE-',
            #         row_height=35,
            #         tooltip='This is a table')],
            [sg.Table(values = list_data, headings = headings_data, max_col_width = 25, auto_size_columns = True, display_row_numbers = False,
                justification = 'left', num_rows = 5, #alternating_row_color = 'lightblue',
                key = 'Test Table', row_height = 47)]
        ]

        self.tab3 = [
            [sg.Radio('None',                 group_id=0,    key='None',                     default=False),
             sg.Radio('Bitwise',              group_id=0,    key='bitwise',                  default=False),
             sg.Radio('Overlay',              group_id=0,    key='Overlay',                  default=False)],
            [sg.CBox('Load Image',            size=(9,1),    key='imload',                   default=False)],               #SIZE?
            [sg.Text('Image Select',          size=(9,1)),
             sg.Slider(range=(1, 10),         size=(20,15),  key='imsave',                   default_value=1,     orientation='h')],
            [sg.Frame('Camera Settings',[
            [sg.Text('Save and exit to apply settings')],
            [sg.Text('FPS',                   size=(9,1)),
             sg.Input(self.config['FPS'],     size=(9,1),    key='FPS')],
             #sg.InputText(self.config['FPS'], size=(9,1),  key='FPS')],
            [sg.Text('Exposure',              size=(9,1)),
             sg.Input(                        size=(9,1),    key='Exposure',                 default_text=self.config['Exposure'],)],
            [sg.Text('Gain',                  size=(9,1)),
             sg.Input(                        size=(9,1),    key='Gain',                     default_text=self.config['Gain'])]])]
        ]
        
        self.col = [
            [sg.Text('Window Size:'),
             sg.Slider(range=(10, 100),      size=(34,10),  key='Window Size',               default_value=self.config['Window Size'],     orientation='h')],
            [sg.Button('Select Front Edge'),    sg.Button('Select Fixture'),    sg.Button('Capture Image'),    sg.Button('Save and Exit')],
            [sg.TabGroup([[sg.Tab('Operation', self.tab0), sg.Tab('Settings', self.tab1), sg.Tab('Help', self.tab2), sg.Tab('Debugging', self.tab3)]])]            
        ]

        self.layout = [
            [sg.Image(filename="", key='IMG', enable_events=True), sg.Column(self.col)]
        ]
        
        self.window = sg.Window(self.config['Welder ID'], self.layout, location=self.config['Window Location'])

        

 #### MAIN LOOP #####################################################################################################################################################################
    def loop(self):
        self.cycle = timer() - self.start; self.start = timer();
        self.event, self.values = self.window.read(timeout=20)
    #CHECK EVENTS AND PERFORM ASSOCIATED FUNCTION
        if self.event == sg.WIN_CLOSED           : return False#os._exit(0)
        if self.event == 'Save and Exit'         : self.save_and_exit({'LOCATION':self.window.CurrentLocation()})
        if self.event == 'Capture Image'         : pickle.dump(self.cam.get_image(), open('image{}.txt'.format(str(int(self.values['imsave']))), 'wb'))
        if self.event == 'Select Front Edge'     :
            self.image.overlay *= False
            cv2.putText(self.image.overlay,'Click and drag to draw box, place center of cross at center of front edge',(150,self.image.height-64),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,255,255),1)
            self.config.update({'Electrode ROI': self.image.select_region(self.window.CurrentLocation(), bounds = False)})
        if self.event == 'Select Fixture'        :
            self.image.overlay *= False
            cv2.putText(self.image.overlay,'Click and drag to draw box around Fixture Square',(150,self.image.height-64),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,255,255),1)
            self.config.update({'Fixture Region': self.image.select_region(self.window.CurrentLocation(), bounds = True)})
        

    #DEFINE VARAIABLES TO AVOID CONSTANT READS FROM VALUES DICTIONARY
        self.fixture_region_visible   = self.values['Fixture Region Visible']
        self.fixture_features_visible = self.values['Fixture Features Visible']
        self.fixture_threshold_values = (self.values['Fixture h'], self.values['Fixture s'], self.values['Fixture v'], self.values['Fixture H'], self.values['Fixture S'], self.values['Fixture V'])
        self.fixture_offset           = (self.values['X Offset']*(-1 if self.values['Fixture Reverse X'] else 1), self.values['Y Offset']*(-1 if self.values['Fixture Reverse Y'] else 1))
        self.contour_size             = (self.values['Contour Min'], self.values['Contour Max'])
        self.highlight_size           = (int(self.values['Highlight Width']), int(self.values['Highlight Height']))
        self.gap_box_size             = (int(self.values['Gap Box Width']), int(self.values['Gap Box Height']))
        self.gap_threshold            = self.values['Gap Light Threshold']
        self.gap_size_limit           = self.values['Gap Size Limit']
        self.fixture_region           = self.config['Fixture Region']
        self.electrode_roi            = self.config['Electrode ROI']

        if self.values['imload']: #LOAD SAVED IMAGE IF ENABLED IN GUI
            try:
                self.image = Frame(pickle.load(open('image{}.txt'.format(str(int(self.values['imsave']))),'rb')))
            except:
                None    #self.image = Frame(self.cam.get_image())#Get image from camera and pre-process, creates image object with layers as properties
        else:
            self.image = Frame(self.cam.get_image())
            
        cv2.putText(self.image.overlay, 'Cycle:%.3f'%self.cycle, (30,self.image.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)#display cycle time
        self.image.blurred = self.image.blur(self.image.img, int(self.values['Blur Value']), (int(2*self.values['Kernel']+1)))

        if self.values['Fixture Enable']:
            fixture_point = self.image.locate_fixture(self.image.blurred, self.fixture_region, self.fixture_threshold_values, self.contour_size, self.fixture_region_visible, self.fixture_features_visible)
            if fixture_point: self.electrode_roi = (fixture_point[0] + fixture_offset[0], fixture_point[1] +  fixture_offset[1])

        self.image.draw_highlight(self.gap_threshold, self.electrode_roi, self.highlight_size, self.gap_box_size, self.gap_size_limit)

        self.config['Output'] = (self.values['Fixture HSV']*'Fixture HSV'+self.values['GAP HSV']*'GAP HSV'+self.values['Overlay']*'Overlay'+self.values['bitwise']*'bitwise') #SUM OF PRODUCTS, ONLY 1 SHOULD REMAIN
        self.image.overlay *= not self.values['Hide Overlay'] #HIDE OVERLAY
        self.window['IMG'].update(data=self.image.output(self.values['Window Size'],self.config['Output'])) #SEND IMAGE TO WINDOW
        return True

    def save_and_exit(self, location):
        self.config.update(location)
        self.config.update(self.values)
        pickle.dump(self.config, open('{}Psave.txt'.format(self.config['Welder ID']), 'wb'))
        self.cam.exit()
        self.window.close()
        os._exit(0)

    def exit_unsaved(self):
        self.cam.exit()

class Frame:   
    def __init__(self, array):
        self.img         = array
        self.img         = cv2.cvtColor(self.img,cv2.COLOR_BayerRG2RGB)
        #ROTATION NO LONGER USED#self.img         = rotation * np.rot90(self.img,3)#Rotates image 90 degrees
        self.img         = cv2.pyrDown(self.img)#scales image down 1/4
        self.height      = self.img.shape[0]
        self.width       = self.img.shape[1]
        self.empty       = np.zeros_like(self.img)#initializing empty arrays
        self.overlay     = np.zeros_like(self.img)#
        self.cannyThresh = np.zeros_like(self.img)#
        self.houghThresh = np.zeros_like(self.img)#

    def output(self, *args):
        try:
            if args[1]   == 'Fixture HSV': self.out = self.fixture_threshold_image
            elif args[1] == 'GAP HSV'   : self.out = self.gap_threshold_image
            elif args[1] == 'Overlay'   : self.out = self.overlay
            elif args[1] == 'bitwise'   : self.out = cv2.bitwise_or(self.img, self.overlay)
            else                        : self.out = np.where(self.overlay == 0, self.img, self.overlay)
            self.out = cv2.resize(self.out,(int(self.out.shape[1]*args[0]/100),int(self.out.shape[0]*args[0]/100)))
            self.imgbytes = cv2.imencode('.png',self.out)[1].tobytes()
            return self.imgbytes
        except: raise(Exception)

    def blur(self, array, blur, kernel):
        _out = cv2.GaussianBlur(array, (kernel,kernel), blur)
        return _out

    def morph_close(self, array, kernel):
        _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel,kernel))
        _out = cv2.morphologyEx(array, cv2.MORPH_CLOSE, _kernel, 3)
        return _out

    def thresh(self, array, threshold):
        h, s, v, H, S, V = threshold
        _out = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
        _out = cv2.inRange(_out, (h,s,v),(H,S,V))
        return _out

    def select_region(self, loc, bounds):
        cv2.putText(self.img,'SELECT WITH MOUSE, SPACEBAR TO CONFIRM',(50,self.height-24),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(1,255,255),2) #WRITE INSTRUCTIONS ONTO IMAGE
        cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('ROI', loc[0], 0) #MOVE TO CURRENT WINDOW LOCATION
        x, y, w, h = cv2.selectROI('ROI', cv2.bitwise_or(self.img, self.overlay)) #CALL ROI SELECTION AND SAVE RESULTS
        cv2.destroyAllWindows()
        if not bounds: #RETURN CENTER OF ROI, ELSE RETURN CORNER POINTS TO USE IN RECT
            return (int(x+w/2), int(y+h/2))
        return (int(x),int(x+w),int(y),int(y+h))

    def locate_fixture(self, array, fixture_region, fixture_threshold_values, contour_size, fixture_region_visible, fixture_features_visible):
        #FINDS BLOBS USING HUE/SATURATION/VALUE AND SIZE CONSTRAINTS, FOR LARGEST RETURNS CENTERPOINT IF FOUND
            self.fixture_threshold_image = self.thresh(array, fixture_threshold_values) #THRESHOLD IMAGE USING HSV
            x_1, x_2, y_1, y_2 = fixture_region #DEFINE REGION BOUNDS
            _contours, _hierarchy = cv2.findContours(self.fixture_threshold_image[y_1:y_2,x_1:x_2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #FIND COUNTOURS IN THRESHOLD
            if fixture_region_visible: #DRAW REGION IF ENABLED IN GUI
                cv2.rectangle(self.overlay, (x_1, y_1), (x_2, y_2), (36, 255, 12), 2) #DRAW RECTANGLE AROUND SEARCH REGION
            if not _contours: #RETURN IF NO CONTOURS FOUND
                return False
            if fixture_features_visible: #DRAW COUNTOUR BOUNDS AND LABEL AREAS IF ENABLED IN GUI
                cv2.drawContours(self.overlay[y_1:y_2,x_1:x_2], _contours, -1, (36, 255, 12), 2) #DRAW ALL CONTOURS
                _sorted_contour_list = [(_contour, str(int(cv2.contourArea(_contour)/100))) for _contour in _contours if cv2.contourArea(_contour) > 100] #CREATE LIST OF TUPLES (CONTOUR, STR(AREA)) FOR CONTOURS OVER 100
                for _contour in _sorted_contour_list: #LABEL EACH CONTOUR FROM SORTED LIST
                    x, y, w, h = cv2.boundingRect(_contour[0]) #GET BOUNDING RECTANGLE AROUND CONTOUR
                    _contour_string_position = (int(x_1 + x + w/2),int(y_1 + y + h + 5)) #CALCULATE POSITION UNDERNEATH CONTOUR
                    cv2.putText(self.overlay, _contour[1], _contour_string_position, cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2) #LABEL AREA BELOW CONTOUR RECT
            _contour_area_list = [cv2.contourArea(contour) for contour in _contours] #LIST OF AREAS FOR EACH CONTOUR
            _max_index = _contour_area_list.index(max(_contour_area_list)) #INDEX OF LARGEST CONTOUR (BEST CANDIDATE FOR FIXTURE POINT)
            _fixture_area = _contour_area_list[_max_index] #AREA OF LARGEST CONTOUR
            _fixture_contour = _contours[_max_index] #CONTOUR OF PROPOSED FIXTURE POINT
            if not contour_size[0] < _fixture_area < contour_size[1]: #RETURN NOTHING IF FIXTURE POINT OUTSIDE SIZE RANGE SELECTED IN GUI
                return False
            x, y, w, h = cv2.boundingRect(_fixture_contour) #BOUNDING RECTANGLE OF CONTOUR
            _fixture_point_x, fixture_point_y = int(x_1 + x + w/2), int(y_1 + y + h/2) #CALCULATE FIXTURE CENTERPOINT
            cv2.putText(self.overlay, 'Fixture:({},{}),  {:.0f}'.format(_fixture_point_x,_fixture_point_y,area/100), (600,self.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2) #LABEL FIXTURE POINT POSITION BOTTOM OF WINDOW
            cv2.circle(self.overlay, (_fixture_point_x,_fixture_point_y), 10, (255,36,255), -1) #DRAW CIRCLE AT FIXTURE CENTERPOINT
            return (_fixture_point_x,_fixture_point_y)

    def draw_highlight(self, gap_threshold, electrode_roi, highlight_size, gap_box_size, gap_size_limit):
            w, h = gap_box_size #WIDTH AND HEIGHT FOR GAP CHECKING BOX
            x, y = electrode_roi #CENTERPOINT OF REGION OF INTEREST (ELECTRODE/FIXTURE EDGE)
            a, b = highlight_size #WIDTH AND HEIGHT TO HIGHLIGHT GAP PIXELS
            self.gap_threshold_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV) #CONVERT TO HSV
            self.gap_threshold_image = cv2.inRange(self.gap_threshold_image, (0,0,0), (180,255,gap_threshold)) #THRESHOLD TO SELECT ANY PIXELS DARKER THAN THRESHOLD
            self.gap_area = np.sum(np.true_divide(self.gap_threshold_image[y-h:y+h, x-w:x+w], 255)) #SUM REMAINING PIXELS INSIDE BOX REGION
            self.highlight_image = np.zeros_like(self.empty) #CREATE EMPTY IMAGE
            if self.gap_area < gap_size_limit: #IF PIXEL SUM LESS THAN LIMIT TURN IMAGE GREEN
                self.highlight_image[:,:,:] = (0,255,0) #GREEN IMAGE
                cv2.putText(self.overlay, 'OK'.format(self.gap_area), (500,70), cv2.FONT_HERSHEY_DUPLEX, 3, (1,255,1),5) #PRINT 'OK' AT TOP OF SCREEN FOR OPERATOR
            else: #IF PIXEL SUM GREATER THAN LIMIT TURN IMAGE RED
                self.highlight_image[:,:,:] = (0,0,255) #RED IMAGE
                cv2.putText(self.overlay, 'CHECK', (450,70), cv2.FONT_HERSHEY_DUPLEX, 3, (1,255,255),5) #PRINT 'CHECK' AT TOP OF SCREEN FOR OPERATOR TO ADJUST ELECTRODE
            self.highlight_image = cv2.bitwise_and(self.highlight_image,self.highlight_image, mask=self.gap_threshold_image) #CUT GAP PIXELS OUT OF RED/GREEN IMAGE
            self.overlay[y-b:y+b, x-a:x+a] = np.where(self.highlight_image[y-b:y+b, x-a:x+a] == 0, self.overlay[y-b:y+b, x-a:x+a], self.highlight_image[y-b:y+b, x-a:x+a]) #DRAWS RED/GREEN HIGHLIGHT ONTO DISPLAYED IMAGE
            cv2.putText(self.overlay, 'Gap:{:.0f}'.format(self.gap_area), (300,self.height-24), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2) #PRINT GAP COUNT AT BOTTOM OF SCREEN
            cv2.rectangle(self.overlay, (x-w, y-h), (x+w, y+h), (255,1,1),3) #DRAW RECTANGLE AROUND GAP CHECK AREA (ELECTRODE ROI + BOX)
    
class Camera: #INITIALIZE SELECTED CAMERA, SET CAMERA PROPERTIES (STREAM BUFFER, FPS, EXPOSURE, GAIN)
    def __init__(self, cam_id, Gain, Exposure, FPS):
        self.system = PySpin.System.GetInstance()
        self.list = self.system.GetCameras()
        self.cam = self.list.GetBySerial(cam_id)
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
        print('\nExposure = {:.2f}\tGain = {:.2f}\tFPS = {:.2f}'.format(PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('Gain')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate')).GetValue()),end='\n')
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


def welder_selector(cameras, welders):
    DEBUG = False
    _n_cams = len(cameras)
    _n_weld = len(welders)
    if _n_cams == 0 and DEBUG:
        return 'Percussion Welder #2'
    if _n_weld == 0 and not DEBUG:
        sg.popup_timed('\nNo cameras detected\n\nCheck camera connection and try again\n',auto_close_duration=3)
        os._exit(0)
        return None
    if _n_weld == 1:
        return welders[0]
    if _n_weld >= 2:
        sg.theme('DarkAmber')
        config_layout = [
                    [sg.Text('Select an available welder:',size=(30, 1),   justification='left', font = ("Helvetica", 13, "") ),],
                    [[sg.Text('',size=(5,2)), sg.Button(welders[i])] for i in range(len(welders))],
                    [sg.Text('', size=(1,2)), sg.CBox('Reset Window Location', key='Reset Window Location',default=False, tooltip='Use if windows launch offscreen')],
                    [sg.Text('', size = (5,2)), sg.Submit(), sg.Text('', size=(2,1)), sg.Cancel()]]
        window = sg.Window('Welder selection', config_layout)    
        _event, _values = window.read(close=True)
        reset_window = _values['Reset Window Location']
        print(f'w {reset_window}, e {_event}')
        if event == sg.WIN_CLOSED   : os._exit(0)
        if event == 'Cancel'        : os._exit(0)
        if event == 'Submit'        : return reset_window, welders[0]  #FIRST AVAILABLE WELDER
        else                        : return reset_window, _event      #RETURN SELECTED WELDER

def startup():#CHECKS CONNECTED CAMERAS, LOADS OR GENERATES SETTINGS    
    welders = {'Percussion Welder #1':'20227035','Percussion Welder #2':'20348005','Percussion Welder #3':'20347994'}    
    parameters = {'FPS':15,'Exposure':12000,'Gain':20}#DEFAULT SETTINGS
    cameras = []
    reset_window = False
    open_windows = get_window()    
    system = PySpin.System.GetInstance()
    camera_list = system.GetCameras()
    DEBUG = True
    for i in range(camera_list.GetSize()):
         camera_list.GetByIndex(i).Init()
         cameras.append(PySpin.CStringPtr(camera_list.GetByIndex(i).GetNodeMap().GetNode('DeviceID')).GetValue())
         camera_list.GetByIndex(i).DeInit

    active_welders    = [k for k, v in welders.items() if k in open_windows]
    available_welders = [k for k, v in welders.items() if v in cameras and k not in open_windows]
    #available_welders = welders
    #available_welders = ['Percussion Welder 1', 'Percussion Welder 2', 'Percussion Welder 3']
    #print(f'w {welders}, wi {welders.keys[0]}')

    selected_welder = welder_selector(cameras, list(available_welders))
    print(f'{selected_welder}')
    
    # if len(cameras) == 0 and not DEBUG:#NO CAMERAS DETECTED, RETURN ERROR
    #     return
                 

    # if len(cameras) ==0 and DEBUG:
    #     selected_welder = 'Percussion Welder #2'

    # if len(cameras) >= 1 and len(available_welders) == 0:#ALL CAMERAS IN USE, RETURN ERROR
    #     sg.popup_timed('\n\n\nAll connected cameras are already in use\n\nPlease close a window before relaunching Autowelder\n\n',auto_close_duration=4)
    #     os._exit(0)
    
    # if len(cameras) == 1 and len(available_welders) == 1:
    #     #sg.popup_timed('\n\nOne camera detected\nLaunching {}\n\n'.format(available_welders[0]),auto_close_duration=2)
    #     _, values = sg.Window('Autowelder',[[sg.Text('\nOne camera detected\nLaunching {}'.format(available_welders[0]),s=(35,3))],[sg.CBox('\nReset Window Location\n',key='RESET LOCATION',default=False,tooltip='Use if windows launch offscreen')],[sg.OK(s=10)]]).read(close=True,timeout=4000)
    #     reset_location = values['RESET LOCATION']
    #     selected_welder = available_welders[0]

    # #MULTIPLE CAMERAS CONNECTED                       
    # if len(cameras) >= 2:

    #     #AUTOMATICALLY SELECT CAMERA IF ONLY 1
    #     if len(available_welders) == 1:
    #         _, values = sg.Window('Autowelder',[[sg.Text('\nOne or more welders already running\nLaunching available {}'.format(available_welders[0]),s=(35,3))],[sg.CBox('\nReset Window Location\n',key='RESET LOCATION',default=False,tooltip='Use if windows launch offscreen')],[sg.OK(s=10)]]).read(close=True,timeout=4000)
    #         reset_location = values['RESET LOCATION']
    #         selected_welder = available_welders[0]

    #     #PROMPT SELECTION IF MULTIPLE CAMERAS            
    #     elif len(available_welders) >= 2:
    #         config_layout = [
    #             [sg.Text('Choose a welder:',size=(30, 1),   justification='left'), ],
    #             [sg.Text('',size=(2,3)),sg.Listbox(values=available_welders,size=(25, 4), key='WELDER ID'),sg.Text('',size=(2,6))],
    #             [sg.CBox('Reset Window Location',key='RESET LOCATION',default=False, tooltip='Use if windows launch offscreen')],
    #             [sg.Submit(), sg.Cancel()]]
    #         window = sg.Window('Welder selection', config_layout)    
    #         event, values = window.read(close=True)
    #         if event == sg.WIN_CLOSED   : os._exit(0)
    #         if event == 'Cancel'        : os._exit(0)
    #         if values['WELDER ID'] == []: values['WELDER ID'] = available_welders[0]#EMPTY SUBMIT SELECTS FIRST WELDER
    #         reset_location = values['RESET LOCATION']
    #         selected_welder = values['WELDER ID'][0]
    #print(f'Window: {reset_window}')

    selected_camera = welders[selected_welder]
    save_file = '{}Psave.txt'.format(selected_welder)
    
    #LOAD OR PROMPT SETTINGS FOR SELECTED CAMERA
    if os.path.isfile(save_file):
        parameters = pickle.load(open(save_file,'rb'))
    else                        :
        v = camera_setup(parameters, selected_welder)
        parameters.update(v)
    
    if reset_window: parameters.update({'Window Location':(0,0)})
    parameters.update({'CAM ID':selected_camera, 'Welder ID':selected_welder})
    camera_list.Clear()
    return parameters

def camera_setup(params, welder):#SET UP CAMERA ROTATION, FPS, EXPOSURE, GAIN
    print('No settings found, starting from scratch...')
    config_layout = [
        [sg.Text('Save file not found\nChoose settings for {}:'.format(welder),justification='left'),],
        [sg.Frame('Camera Settings',[
        [sg.Text('FPS',size=(25,1)),     sg.InputText(params['FPS'],size=(9,1),     key='FPS')],
        [sg.Text('Exposure',size=(25,1)),sg.InputText(params['Exposure'],size=(9,1),key='Exposure')],
        [sg.Text('Gain',size=(25,1)),    sg.InputText(params['Gain'],size=(9,1),    key='Gain')],
        [sg.CBox('Reset Window Location', key='Reset Window Location',default=False, tooltip='Use if windows launch offscreen')],
        ])],
        [sg.Submit(), sg.Cancel()]]
    _event, _values = sg.Window('Camera Setup', config_layout).read(close=True)
    if _values['Reset Window Location']: _values.update({'Window Location':(0,0)})
    if _event == sg.WIN_CLOSED : os._exit(0)
    if _event == 'Cancel'      : os._exit(0)
    return _values
    
def main():
    loop = True
    #SETS UP CAMERA EXPOSURE, GAIN, FPS
    settings = startup()
    #CREATES APPLICATION GUI WINDOW INSTANCE
    win = WelderInstance(settings)
    #INITIALIZE SELECTED CAMERA WITH SETTINGS
    win.cam = Camera(settings['CAM ID'],int(settings['Gain']),int(settings['Exposure']),int(settings['FPS']))

    #MAIN LOOP, GRABS AND PROCESSES CAMERA IMAGES, RESPONDS TO USER INPUTS
    while loop:
        loop = win.loop()        

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()
