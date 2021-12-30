import math, base64, pickle, PySpin, time, threading, queue
import sys, os, cv2, re, pdb
import PySimpleGUI as sg
import numpy as np
from timeit import default_timer as timer

#initialized_vars = pickle.load(open('pGUI.txt.','rb'))
sg.theme('Reds')
# Define the window layout-----------------------------------------------------------------------------------------------

#Layout for Camera 0~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~
class Instance: #Creates pyGUI window for each camera instance
    def __init__(self, settings):
        self.welds = [[0,0,0]]
        self.weld_target = [[0,0]]
        self.cam = None
        self.start = 0
        self.config= {'wRmin':50,'SQUARE LOW H':0,'SQUARE HIGH H':180,'SQUARE LOW S':0,'SQUARE HIGH S':255,'SQUARE LOW V':0,'SQUARE HIGH V':255,'SQUARE CANNY A':100,'SQUARE CANNY B':200,'SQUARE CANNY C':1,'SQUARE X':0,'SQUARE Y':0,'sq_x1':600,'sq_x2':900,'sq_y1':600,'sq_y2':900,'Area Height':200,'Area Width':200,'Box Height':40,'Box Width':70,'Gap Threshold':100,'Light Threshold':50,'cMin':100,'cMax':1500,'Fixture':False,
                      'wRmax':100,'Legacy':True,'Exposure':20000,'Gain':20,'re_setting':True,'Output':'None','t_min':1,'t_max':255,'CANNY V':31,'morph':1,'morph kernel':1,'dilate':1,'h_lineThresh':50,'v_lineThresh':50,'KERNEL':10,'LOCATION':(10,100),'ix':279,'iy':286,'ox':290,'oy':295,'icirr':82,'ocirr':147,'rx1':262,'rx2':692,'ry1':423,'ry2':793,'region_selected':True,'LOW H':6,'HIGH H':26,'LOW S':0,'HIGH S':255,'LOW V':0,'HIGH V':255,'thresh_val':84,'e_x':770,'e_y':10,'e_w':96,'e_h':59,'w_x':114,'w_y':301,'w_w':472,'w_h':440,'v_x1':709,'v_y1':419,'v_x2':866,'v_y2':671,'h_x1':340,'h_y1':273,'h_x2':663,'h_y2':441,'RESIZE':50.0,'CIRCLES':True,'OUTLINE':False,'ALL CIRCLES':False,'BOXES':True,'EDGES':True,'Weld Status':True,'RCenter':True,'X Shift':50.0,'Y Shift':50.0,'Th':10.0,'0':True,'THRESH':False,'THRESH SLIDER MIN':28.0,'THRESH SLIDER MAX':255.0,'CANNY':False,'CANNY A':7.0,'CANNY B':37.0,'BLUR':False,'BLUR VAL':5.0,'HSV':False,'CANNY LOW H':6.0,'CANNY HIGH H':26.0,'CANNY LOW S':0.0,'CANNY HIGH S':255.0,'CANNY LOW V':0.0,'CANNY HIGH V':255.0,'HOUGH LOW H':6.0,'HOUGH HIGH H':26.0,'HOUGH LOW S':0.0,'HOUGH HIGH S':255.0,'HOUGH LOW V':0.0,'HOUGH HIGH V':255.0,'V VECTOR':213.0,'H VECTOR':316.0,'minR':52.0,'maxR':137.0,'minDist':1,'PARAM1':1,'PARAM2':20.0,'PARAM3':15.0,'lineThresh':0,'minLineLength':50,'maxLineGap':15,}
        
        if settings is not None: self.config.update(settings)
        
        self.tab1 =  [
            [sg.Frame(layout=[
            [sg.CBox('Show Weld Center Guess',    key='CIRCLES',      default=True,)],
            [sg.CBox('Show Outer Weld Bound',     key='OUTLINE',      default=False,)],
            
            [sg.CBox('Show Boxes',                key='BOXES',        default=True)],
            [sg.CBox('Show Labels',                key='Labels',        default=True)],
            [sg.CBox('Show Edges',                key='EDGES',        default=False)],
            [sg.CBox('Show Weld Judgement',       key='Weld Status',  default=False)],
            #[sg.CBox('Show Weld Reset Center',    key='RCenter',  default=True)],
            [sg.CBox('Hide All',         key='HIDE ALL',  default=False)],
            ], title='Options', relief=sg.RELIEF_SUNKEN)],
            #[sg.Text('width'),sg.Slider((1, 300),215,1,orientation='h',size=(40, 15),key='width',),],
            #[sg.Text('height'),sg.Slider((1, 300),200,1,orientation='h',size=(40, 15),key='height',),],
            #[sg.Text('Vertical Offset'),sg.Slider((1, 300),213,1,orientation='h',size=(40, 15),key='v_offset',),],
            #[sg.Text('Horizontal Offset'),sg.Slider((1, 300),286,1,orientation='h',size=(40, 15),key='h_offset',),],
            #[sg.Text('Proportion'),sg.Slider((1, 100),25,1,orientation='h',size=(40, 15),key='porp',),],
            #[sg.Text('V/N'),sg.Slider((100, 500),200,1,orientation='h',size=(20, 15),key='V VICTOR',),sg.Slider((100, 500),200,1,orientation='h',size=(20, 15),key='V NORMAL')],
        ]


        self.tab2 = [
            [sg.Radio('None', 'Radio', False, size=(10, 1)),
             sg.Radio('Canny', 'Radio', False, size=(10, 1), key='CANNY'),
             sg.Radio('Blur', 'Radio', False, size=(10,10), key='CANNY BLUR')],
            [sg.Radio('Threshold', 'Radio', True, size=(10,1), key='CANNY HSV'),
             sg.CBox('Legacy',    key='Legacy',      default=self.config['Legacy'],)],
            
            #[sg.Text('BW Thresh',size=(9,1)),
            # sg.Slider((0, 255),self.config['THRESH SLIDER MIN'],1,orientation='h',size=(20, 15),key='THRESH SLIDER MIN',),
            # sg.Slider((0, 255),self.config['THRESH SLIDER MAX'],1,orientation='h',size=(19, 15),key='THRESH SLIDER MAX',),],
            [sg.Text('Canny A/B',size=(9,1)),
             sg.Slider((1, 500),self.config['CANNY A'],1,orientation='h',size=(20, 15),key='CANNY A',),
             sg.Slider((1, 500),self.config['CANNY B'],1,orientation='h',size=(19, 15),key='CANNY B',),],
            #[sg.Text('Morphs',size=(9,1)),
            # sg.Slider((1, 11),self.config['morph'],1,orientation='h',size=(20, 15),key='morph',),
            # sg.Slider((1, 10),self.config['morph kernel'],1,orientation='h',size=(19, 15),key='morph kernel',)],
            #[sg.Slider((1, 11),self.config['dilate'],1,orientation='h',size=(40, 15),key='dilate',)],
            [sg.Text('Blur/Kernel',size=(9,1)),
             sg.Slider((0, 11),self.config['BLUR VAL'],1,orientation='h',size=(20, 15),key='BLUR VAL',),
             sg.Slider((1, 10),self.config['KERNEL'],1,orientation='h',size=(19, 15),key='KERNEL',)],

            [sg.Frame('Line Detection',[
            [sg.Text('FLD Length',size=(9,1)),sg.Slider((1, 100),10,1,orientation='h',size=(40, 15),key='f minLineLength',),],            
            [sg.Text('FLD Gap',size=(9,1)),sg.Slider((1, 100),14,1,orientation='h',size=(40, 15),key='f maxLineGap',)]])],
            

            [sg.Frame('Vertical',[
            [sg.Text('V Threshold',size=(9,1)),sg.Slider((1, 200),self.config['v_lineThresh'],1,orientation='h',size=(40, 15),key='v_lineThresh',),],
            [sg.Text('V Gap',size=(9,1)),sg.Slider((1, 50),15,1,orientation='h',size=(40, 15),key='v maxLineGap')],
            [sg.Text('V Length',size=(9,1)),sg.Slider((1, 1000),50,1,orientation='h',size=(40, 15),key='v minLineLength')]])],

            [sg.Frame('Horizontal',[
            [sg.Text('H Threshold',size=(9,1)),sg.Slider((1, 100),self.config['h_lineThresh'],1,orientation='h',size=(40, 15),key='h_lineThresh',),],
            [sg.Text('H Gap',size=(9,1)),sg.Slider((1, 50),15,1,orientation='h',size=(40, 15),key='h maxLineGap',)],
            [sg.Text('H Length',size=(9,1)),sg.Slider((1, 1000),50,1,orientation='h',size=(40, 15),key='h minLineLength')]])],

            
            #[sg.Text('HUE',size=(9,1)),
            # sg.Slider((0, 180),self.config['CANNY LOW H'],1,orientation='h',size=(20, 15),key='CANNY LOW H'),
            # sg.Slider((0, 180),self.config['CANNY HIGH H'],1,orientation='h',size=(20, 15),key='CANNY HIGH H')],
            #[sg.Text('SAT',size=(9,1)),
            # sg.Slider((0, 255),self.config['CANNY LOW S'],1,orientation='h',size=(20, 15),key='CANNY LOW S'),
            # sg.Slider((0, 255),self.config['CANNY HIGH S'],1,orientation='h',size=(20, 15),key='CANNY HIGH S')],
            #[sg.Text('VAL',size=(9,1)),
            # sg.Slider((0, 255),self.config['CANNY LOW V'],1,orientation='h',size=(20, 15),key='CANNY LOW V'),
            # sg.Slider((0, 255),self.config['CANNY HIGH V'],1,orientation='h',size=(20, 15),key='CANNY HIGH V')],
        ]
        
        self.tab3 =  [
            [sg.Text('These are the settings used for weld detection, do not touch anything')],        
            [sg.Radio('None', 'Radio', True, size=(10, 1)),
             sg.Radio('Blur', 'Radio', size=(10, 1), key='HOUGH BLUR'),
             sg.Radio('Threshold', 'Radio', size=(10, 1), key='HOUGH HSV')],
            [sg.CBox('Detected Circles',        key='ALL CIRCLES',  default=False)],
             

            #[sg.Text('Threshold',size=(10,1)),
            # sg.Slider((0, 255),self.config['THRESH SLIDER MIN'],1,orientation='h',size=(20, 15),key='THRESH SLIDER MIN',),
            # sg.Slider((0, 255),self.config['THRESH SLIDER MAX'],1,orientation='h',size=(20, 15),key='THRESH SLIDER MAX',)],

            [sg.Frame('HSV',[[        
             sg.Text('HUE',size=(9,1)),
             sg.Slider((0, 180),self.config['HOUGH LOW H'],1,orientation='h',size=(20, 15),key='HOUGH LOW H'),
             sg.Slider((0, 180),self.config['HOUGH HIGH H'],1,orientation='h',size=(20, 15),key='HOUGH HIGH H')],
            [sg.Text('SAT',size=(9,1)),
             sg.Slider((0, 255),self.config['HOUGH LOW S'],1,orientation='h',size=(20, 15),key='HOUGH LOW S'),
             sg.Slider((0, 255),self.config['HOUGH HIGH S'],1,orientation='h',size=(20, 15),key='HOUGH HIGH S')],
            [sg.Text('VAL',size=(9,1)),
             sg.Slider((0, 255),self.config['HOUGH LOW V'],1,orientation='h',size=(20, 15),key='HOUGH LOW V'),
             sg.Slider((0, 255),self.config['HOUGH HIGH V'],1,orientation='h',size=(20, 15),key='HOUGH HIGH V')]])],

            
                                       
            [sg.Frame('Weld Detection',[[
             sg.Text('Weld Radius',size=(9,1)),
             sg.Slider((10, 100),self.config['minR'],1,orientation='h',size=(20, 15),key='minR'),
             sg.Slider((50, 300),self.config['maxR'],1,orientation='h',size=(20, 15),key='maxR')],
            [sg.Text('Distance',size=(9,1)),sg.Slider((1, 300),self.config['minDist'],1,orientation='h',size=(40, 15),key='minDist')],                          
            [sg.Text('Threshold',size=(9,1)),sg.Slider((1, 300),self.config['PARAM1'],1,orientation='h',size=(40, 15),key='PARAM1')],
            [sg.Text('Accumulator',size=(9,1)),sg.Slider((1, 100),self.config['PARAM2'],1,orientation='h',size=(40, 15),key='PARAM2')],
            [sg.Text('dp',size=(9,1)),sg.Slider((1, 20),10,1,orientation='h',size=(40, 15),key='dp')]])],

                                    
            #[sg.Text('CANNY V'),sg.Slider((1, 100),self.config['CANNY V'],1,orientation='h',size=(40, 15),key='CANNY V')],
        ]


        self.tab4 = [
            [sg.Frame('Weld Target',[
            [sg.Text('V',size=(4,1)),
             sg.Slider((0, 500),self.config['V VECTOR'],1,orientation='h',size=(20, 15),key='V VECTOR',),
             sg.Text('  H',size=(4,1)),
             sg.Slider((0, 500),self.config['H VECTOR'],1,orientation='h',size=(20, 15),key='H VECTOR')],
            [sg.Text('Radius',size=(4,1)),
             sg.Slider((0, 100),self.config['wRmin'],1,orientation='h',size=(20, 15),key='wRmin',),
             sg.Slider((100, 200),self.config['wRmax'],1,orientation='h',size=(20, 15),key='wRmax')]])],

        ]

        self.tab5 = [      
            [sg.Radio('None', 'Radio', True, size=(10, 1)),
             sg.Radio('Fixture HSV', 'Radio', size=(10, 1), key='SQUARE HSV'),
             sg.Radio('Gap HSV', 'Radio', size=(10, 1), key='GAP HSV'),
             sg.Radio('Misc', 'Radio', size=(10, 1), key='SQUARE MISC')],
            [sg.CBox('Fixture Point',    key='Fixture',      default=self.config['Fixture'])],
    
            [sg.Frame('Offset',[
            ])],



            [sg.Frame('Fixture',[
            [sg.Text('X/Y Offset',size=(10,1)),
             sg.Slider((0, 500),self.config['SQUARE X'],1,orientation='h',size=(20, 10),key='SQUARE X',),
             sg.Slider((0, 500),self.config['SQUARE Y'],1,orientation='h',size=(20, 10),key='SQUARE Y')],
            [sg.Text('Size',size=(4,1)),
             sg.Slider((0, 200),self.config['cMin'],1,orientation='h',size=(20, 10),key='cMin'),
             sg.Slider((100, 500),self.config['cMax'],1,orientation='h',size=(20, 10),key='cMax')],
            [sg.Text('HUE',size=(4,1)),
             sg.Slider((0, 180),self.config['SQUARE LOW H'],1,orientation='h',size=(20, 10),key='SQUARE LOW H'),
             sg.Slider((0, 180),self.config['SQUARE HIGH H'],1,orientation='h',size=(20, 10),key='SQUARE HIGH H')],
            [sg.Text('SAT',size=(4,1)),
             sg.Slider((0, 255),self.config['SQUARE LOW S'],1,orientation='h',size=(20, 10),key='SQUARE LOW S'),
             sg.Slider((0, 255),self.config['SQUARE HIGH S'],1,orientation='h',size=(20, 10),key='SQUARE HIGH S')],
            [sg.Text('VAL',size=(4,1)),
             sg.Slider((0, 255),self.config['SQUARE LOW V'],1,orientation='h',size=(20, 10),key='SQUARE LOW V'),
             sg.Slider((0, 255),self.config['SQUARE HIGH V'],1,orientation='h',size=(20, 10),key='SQUARE HIGH V')]])],


            [sg.Frame('Gap',[
            [sg.Text('Light Threshold',size=(17,1)),sg.Slider((0, 255),self.config['Light Threshold'],                          1,orientation='h',size=(30, 10),key='Light Threshold')],
            [sg.Text('Gap Threshold',size=(17,1)), sg.Slider((1, 200),self.config['Gap Threshold'],                          1,orientation='h',size=(30, 10),key='Gap Threshold',),],
            [sg.Text('Area Height',size=(17,1)),     sg.Slider((100, 300),self.config['Area Height'],                          1,orientation='h',size=(30, 10),key='Area Height',),],
            [sg.Text('Area Width',size=(17,1)),      sg.Slider((100, 300),self.config['Area Width'],                          1,orientation='h',size=(30, 10),key='Area Width',),],
            [sg.Text('Box Height',size=(17,1)),     sg.Slider((10, 200),self.config['Box Height'],                          1,orientation='h',size=(30, 10),key='Box Height',),],
            [sg.Text('Box Width',size=(17,1)),      sg.Slider((10, 200),self.config['Box Width'],                          1,orientation='h',size=(30, 10),key='Box Width',),]])]
            

            

        ]
        
        self.tab0 = [
            [sg.Text('Debugging')],        
            [sg.Radio('None', 'Radio', False, size=(10, 1), key='none'),
             sg.Radio('Bitwise', 'Radio', False, size=(10, 1), key='bitwise'),
             sg.Radio('Overlay', 'Radio', False, size=(10, 1), key='Overlay'),
            ],
            [sg.CBox('Load Image',                key='imload',  default=False)],

            [sg.Text('Image Select',size=(9,1)),
             sg.Slider((1, 10),1,1,orientation='h',size=(40, 15),key='imsave',)],
            [sg.Text('Sobel',size=(9,1)),sg.Slider((1, 3),1,1,orientation='h',size=(40, 15),key='sobel aperture',),],
            
            [sg.Text('Threshold',size=(10,1)),
             sg.Slider((0, 255),self.config['t_min'],1,orientation='h',size=(20, 15),key='t_min',),
             sg.Slider((0, 255),self.config['t_max'],1,orientation='h',size=(20, 15),key='t_max',)],

            [sg.Frame('Camera Settings',[
            [sg.Text('Save and exit to apply settings')],
            [sg.Text('FPS',size=(9,1)),sg.InputText(self.config['FPS'],size=(9,1),key='FPS')],
            [sg.Text('Exposure',size=(9,1)),sg.InputText(self.config['Exposure'],size=(9,1),key='Exposure')],
            [sg.Text('Gain',size=(9,1)),sg.InputText(self.config['Gain'],size=(9,1),key='Gain')]



            ])]
            #[sg.Text('FPS',size=(9,1)),sg.InputText(self.config['FPS'])],
            #[sg.Text('FPS',size=(9,1)),sg.InputText(self.config['FPS'])],
            #[sg.Text('Save and exit to apply settings',size=(9,1))]
            

        ]
        
        self.col = [
            [sg.Text('Resize Window:'),sg.Slider((1, 100),50,1,orientation='h',size=(40, 15),key='RESIZE',),],
            [sg.Button('Select Horizontal Edge'),sg.Button('Select Vertical Edge')],
            [sg.Button('Select Weld'),sg.Button('Select Front Edge')],
            [sg.Button('Save and Exit'),sg.Button('Capture Image')],
            [sg.TabGroup([[sg.Tab('Operation', self.tab1), sg.Tab('Edge Detection', self.tab2), sg.Tab('Weld Detection', self.tab3),sg.Tab('Weld Target', self.tab4),sg.Tab('Electrode',self.tab5),sg.Tab('Debugging',self.tab0)]])],#, sg.Tab('TESTING', self.tab3)
        ]

        self.layout = [
            [sg.Image(filename="", key='IMG',enable_events=True),sg.Column(self.col)]
            
        ]
    
        self.window = sg.Window(self.config['CAM ID'], self.layout, location=self.config['LOCATION'])

#### MAIN LOOP #####################################################################################################################################################################
    def loop(self, i):
        self.cycle = timer() - self.start; self.start = timer();
        self.event, self.values = self.window.read(timeout=20)
        if self.event == sg.WIN_CLOSED: os._exit(0)
        if self.event == 'Select Weld'           : self.config.update(dict(zip(['rx1','rx2','ry1','ry2'],self.image.select_region())))
        if self.event == 'Select Vertical Edge'  : self.config.update(dict(zip(['v_x1','v_x2','v_y1','v_y2'],self.image.select_region())))
        if self.event == 'Select Horizontal Edge': self.config.update(dict(zip(['h_x1','h_x2','h_y1','h_y2'],self.image.select_region())))
        if self.event == 'Select Front Edge'     : self.image.overlay *= False; self.config.update(dict(zip(['sq_x1','sq_x2','sq_y1','sq_y2'],self.image.select_region())))
        if self.event == 'Save and Exit'         : self.save_and_exit({'LOCATION':self.window.CurrentLocation()})#Saves all settings to file
        if self.event == 'Capture Image'         : pickle.dump(self.cam.get_image(), open('image{}.txt'.format(str(int(self.values['imsave']))), 'wb'))
        #if self.event == 'Draw Region'           : self.config.update(self.image.select_region())#
                
        if not self.values['imload']:
            self.image = Frame(self.cam.get_image())#Get image from camera and pre-process, creates image object with layers as properties
        else:
            try:    self.image = Frame(pickle.load(open('image{}.txt'.format(str(int(self.values['imsave']))),'rb')))
            except: self.image = Frame(self.cam.get_image())
        
        cv2.putText(self.image.overlay, 'Cycle:%.3f'%self.cycle, (30,1000), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)#display cycle time
        #cv2.putText(self.image.overlay, str(i), (30,50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,50,30),3)#display cycle time



            
        
        
        try:
            sqx, sqy = self.image.find_gap(self.image.blur(self.image.img,self.values['BLUR VAL'], (int(2*self.values['KERNEL']+1))),
                self.values['Fixture'],
                self.values['SQUARE LOW H'],self.values['SQUARE LOW S'],self.values['SQUARE LOW V'],
                self.values['SQUARE HIGH H'],self.values['SQUARE HIGH S'],self.values['SQUARE HIGH V'],
                self.config['sq_x1'],self.config['sq_x2'],self.config['sq_y1'],self.config['sq_y2'],
                self.values['SQUARE X'],self.values['SQUARE Y'],
                100*self.values['cMin'],100*self.values['cMax'])
        except:raise(Exception)
        
            
        self.image.highlight(0,0,0,180,255,self.values['Light Threshold'],
                             int(sqx),#+self.values['SQUARE X']),#X GAP CENTER
                             int(sqy),#+self.values['SQUARE Y']),#Y GAP CENTER
                             int(self.values['Box Width']),int(self.values['Box Height']),
                             int(self.values['Area Width']),int(self.values['Area Height']),int(self.values['Gap Threshold']))

        

        

        #self.image.blur(self.image.img, self.values['BLUR VAL'],int(2*self.values['KERNEL']+1))
                                 
        '''self.welds = self.image.find_circles(self.welds, self.values['CIRCLES'],self.values['ALL CIRCLES'],
            self.image.blur,self.values['HOUGH LOW H'],self.values['HOUGH LOW S'],self.values['HOUGH LOW V'],
            self.values['HOUGH HIGH H'],self.values['HOUGH HIGH S'],self.values['HOUGH HIGH V'],
            self.config['rx1'],self.config['rx2'],self.config['ry1'],self.config['ry2'],self.values['dp']/10,
            self.values['minDist'],self.values['PARAM1'],self.values['PARAM2'],self.values['minR'],self.values['maxR'])
       
        if not self.values['Legacy']:
            self.image.cannyThresh = self.image.fld(int(2*self.values['KERNEL']+1), self.values['BLUR VAL'], int(self.values['f minLineLength']), self.values['CANNY A'], self.values['CANNY B'], self.values['f maxLineGap']/10, int(2*self.values['sobel aperture']+1))    
        else:
            self.image.cannyThresh = cv2.GaussianBlur(self.image.img,(21,21),self.values['BLUR VAL'])
            self.image.edges(self.image.cannyThresh, self.values['CANNY A'],self.values['CANNY B'],3)
        
        h_line = self.image.find_edges(self.values['EDGES'],self.config['h_x1'],self.config['h_y1'],self.config['h_x2'],self.config['h_y2'],self.values['h_lineThresh'],self.values['h minLineLength'],self.values['h maxLineGap'])
        v_line = self.image.find_edges(self.values['EDGES'],self.config['v_x1'],self.config['v_y1'],self.config['v_x2'],self.config['v_y2'],self.values['v_lineThresh'],self.values['v minLineLength'],self.values['v maxLineGap'])
        self.weld_target = self.image.find_weld_target(h_line,v_line,self.values['V VECTOR'],self.values['H VECTOR'],self.values['wRmin'],self.values['wRmax'],self.weld_target)
        if self.values['Weld Status'] and self.welds.any() and self.weld_target.any():
            try:
                weldAvg = np.average(self.welds,axis=0,weights=self.welds.any(axis=1)).astype(int)
                targAvg = np.average(self.weld_target,axis=0,weights=self.weld_target.any(axis=1)).astype(int)
                delta = math.sqrt((targAvg[0]-weldAvg[0]-self.config['rx1'])**2 + (targAvg[1]-weldAvg[1]-self.config['ry1'])**2)
                if delta < self.values['wRmin']:
                    cv2.putText(self.image.overlay,'Weld is Good',(5,90),cv2.FONT_HERSHEY_DUPLEX,2,(1,255,1),2)
                elif np.max(self.weld_target) < 1224:
                    cv2.putText(self.image.overlay,'Check Weld',(5,90),cv2.FONT_HERSHEY_DUPLEX,2,(1,255,255),2)
            except:None#raise(Exception)
        if self.values['BOXES']:
            try:
                cv2.rectangle(self.image.overlay,(self.config['rx1'],self.config['ry1']),(self.config['rx2'],self.config['ry2']),(1,255,1),2)#Weld Box
                cv2.rectangle(self.image.overlay,(self.config['v_x1'],self.config['v_y1']),(self.config['v_x2'],self.config['v_y2']),(255,1,1),2)#Vertical Edge Box
                cv2.rectangle(self.image.overlay,(self.config['h_x1'],self.config['h_y1']),(self.config['h_x2'],self.config['h_y2']),(255,1,1),2)#Horizontal Edge Box
                if self.values['Labels']:
                    cv2.putText(self.image.overlay,'Weld',           (self.config['rx1']+1,self.config['ry2']+22)  , cv2.FONT_HERSHEY_DUPLEX,1,(1,1,1),2)#Weld Label
                    cv2.putText(self.image.overlay,'Vertical Edge',  (self.config['v_x1']+1,self.config['v_y2']+22), cv2.FONT_HERSHEY_DUPLEX,1,(1,1,1),2)#V Edge Label
                    cv2.putText(self.image.overlay,'Horizontal Edge',(self.config['h_x1']+1,self.config['h_y1']-5), cv2.FONT_HERSHEY_DUPLEX,1,(1,1,1),2)#H Edge Label
            except:
                None#raise(Exception)
        '''
        self.config['Output'] = (self.values['CANNY']*'CANNY'+self.values['CANNY BLUR']*'BLUR'+self.values['CANNY HSV']*'CANNY HSV'+
                                 self.values['HOUGH HSV']*'HOUGH HSV'+self.values['HOUGH BLUR']*'BLUR'+self.values['SQUARE HSV']*'SQUARE HSV'+self.values['GAP HSV']*'GAP HSV'+
                                 self.values['Overlay']*'Overlay'+self.values['bitwise']*'bitwise')
        self.image.overlay *= not self.values['HIDE ALL']
        self.window['IMG'].update(data=self.image.output(self.values['RESIZE'],self.config['Output']))#,self.values['morph']))
####################################################################################################################################################################################

    def save_and_exit(self, location):
        self.values.update(location)
        self.config.update(self.values)
        pickle.dump(self.config, open('{}save.txt'.format(self.config['CAM ID']), 'wb'))
        self.cam.exit()
        self.window.close()
        os._exit(0)

    def save_and_delete(self):
        return None
               

class Frame:   
    def __init__(self, array):
        self.img         = array
        self.img         = cv2.cvtColor(self.img,cv2.COLOR_BayerRG2RGB)
        self.img         = cv2.pyrDown(self.img)
        self.height      = self.img.shape[0]
        self.width       = self.img.shape[1]
        self.empty       = np.zeros_like(self.img)
        self.overlay     = np.zeros_like(self.img)
        self.cannyThresh = np.zeros_like(self.img)
        self.houghThresh = np.zeros_like(self.img)

    def output(self, *args):
        try:
            if   args[1] == 'CANNY'     : self.out = self.edges
            elif args[1] == 'SQUARE HSV': self.out = self.squareThresh
            elif args[1] == 'GAP HSV'   : self.out = self.gapThresh
            elif args[1] == 'CANNY HSV' : self.out = self.cannyThresh#self.cannyThresh#self.out = cv2.erode(self.cannyThresh,self.kernel,args[3])#
            elif args[1] == 'HOUGH HSV' : self.out = self.houghThresh#cv2.bitwise_or(cv2.merge((self.houghThresh,self.houghThresh,self.houghThresh)), self.overlay)        
            elif args[1] == 'Overlay'   : self.out = self.overlay
            elif args[1] == 'bitwise'   : self.out = cv2.bitwise_or(self.img, self.overlay)
            elif args[1] == 'BLUR'      : self.out = self.blur
            elif args[1] == 'HSV'       : self.out = self.thresh
            else                        : self.out = np.where(self.overlay == 0, self.img, self.overlay)
            self.out = cv2.resize(self.out,(int(self.out.shape[1]*args[0]/100),int(self.out.shape[0]*args[0]/100)))
            self.imgbytes = cv2.imencode('.png',self.out)[1].tobytes()
            return self.imgbytes
        except: raise(Exception)

    def blur(self, array, blur, kernel):
        out = cv2.GaussianBlur(array, (kernel,kernel), blur)
        return out

    def edges(self, array, a, b, c):        
        self.edges = cv2.Canny(array, a, b, None, c, False)

    def vertical_edges(self, array, a, b):
        self.v_edges = cv2.Canny(array, a, b)

    def thresh(self, array, h,s,v,H,S,V,*args):
        out = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
        out = cv2.inRange(out, (h,s,v),(H,S,V))
        return out

    def select_region(self):
        cv2.putText(self.img,'SELECT WITH MOUSE, SPACEBAR TO CONFIRM',(200,1000),cv2.FONT_HERSHEY_DUPLEX,1,(255,1,1),3)
        (x,y,w,h) = cv2.selectROI(cv2.bitwise_or(self.img, self.overlay))
        cv2.destroyAllWindows()
        return [int(x),int(x+w),int(y),int(y+h)]


    def find_gap(self,array,Fixture,h,s,v,H,S,V,x1,x2,y1,y2,xoff,yoff,cmin,cmax):
        self.squareThresh = self.thresh(array, h, s, v, H, S, V)
        if Fixture:
            contours, _hierarchy = cv2.findContours(self.squareThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                #cv2.drawContours(self.overlay, contours, -1, (36,255,12), 2)
                area = cv2.contourArea(contour)
                if area > cmin and area < cmax:
                    x,y,w,h = cv2.boundingRect(contour)
                    cv2.putText(self.overlay, 'Fixture:({},{}),  {:.0f}'.format(int(x+w/2),int(y+h/2),area/100), (600,1000), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)
                    cv2.circle(self.overlay, (int(x+w/2),int(y+h/2)), 3, (255,36,12), 2)
                    cv2.rectangle(self.overlay, (x, y), (x + w, y + h), (255,36,12), 2)
                    return (xoff+x+w/2),(yoff+y+h/2)
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
            cv2.putText(self.overlay, 'Gap:{:.0f}'.format(self.gap_area), (300,1000), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,1),2)
            cv2.rectangle(self.overlay, (x-bw,y-bh), (x+bw,y+bh), (255,1,1),3)
        except:
            raise(Exception)

    def fld(self, kernel, blur, length, gap, a, b, sobel):
        self.cannyThresh = cv2.GaussianBlur(self.img, (kernel, kernel), blur)
        self.cannyThresh = cv2.cvtColor(self.cannyThresh, cv2.COLOR_BGR2GRAY)
        v = np.mean(self.cannyThresh)
        lower=int(max(0,(1.0-0.33)*v))
        upper = int(min(255, (1.0 + 0.33) * v))
        #print('U:{}, L:{}'.format(upper,lower))
        if lower <= 0 or upper <= 0: lower = 111; upper = 222
        fld = cv2.ximgproc.createFastLineDetector(
            _length_threshold    = length,
            _distance_threshold  = gap,
            _canny_th1           = a,
            _canny_th2           = b,
            _canny_aperture_size = sobel,
            _do_merge            = False)
        lines = fld.detect(self.cannyThresh)
        #lines2 = fld.detect(self.houghThresh)
        #self.houghThresh = fld.drawSegments(self.houghThresh, lines2)
        self.cannyThresh = fld.drawSegments(self.cannyThresh, lines)
        self.edges(self.cannyThresh, a, b, sobel)
        return self.cannyThresh

    def find_edges(self, visible, x1, y1, x2, y2, thresh, minLength, maxGap):
        try:            
            lines = cv2.HoughLinesP(self.edges[y1:y2,x1:x2], rho=int(1), theta=np.pi/180,  threshold=int(thresh), minLineLength=minLength, maxLineGap=maxGap)
            if lines is not None:
                lx = [l[0][0] for l in lines]
                linex = lines[lx.index(min(lx))][0]
                for line in lines:
                    if visible:
                        cv2.line(self.overlay[y1:y2,x1:x2], (line[0][0], line[0][1]), (line[0][2], line[0][3]), (1,1,255), 3, cv2.LINE_AA)

                #breakpoint()
                lineavg = np.around(np.average(lines, axis=0)).astype(int)
                cv2.line(self.overlay[y1:y2,x1:x2], (lineavg[0][0], lineavg[0][1]), (lineavg[0][2], lineavg[0][3]), (1,255,1), 3, cv2.LINE_AA)
                cv2.line(self.overlay[y1:y2,x1:x2], (lines[0][0][0], lines[0][0][1]), (lines[0][0][2], lines[0][0][3]), (1,255,255), 3, cv2.LINE_AA)
                cv2.line(self.overlay[y1:y2,x1:x2], (linex[0], linex[1]), (linex[2], linex[3]), (255,255,1), 3, cv2.LINE_AA)
                edge = np.add(lines[0][0],[x1,y1,x1,y1])
                return edge
        except:
            raise(Exception)
        

    def find_circles(self, welds, showAvg, showAll, array, h, s, v, H, S, V, x1, x2, y1, y2, dp, minDist, param1, param2, minR, maxR):
        welds = np.concatenate((welds,[[0,0,0]]),axis=0)
        if len(welds) > 6: welds = np.delete(welds, [0,1,2], axis=0)
        self.houghThresh = self.thresh(array, h, s, v, H, S, V)
        self.houghThresh = np.multiply(self.houghThresh,5)
        try:#TRY cv2.HOUGH_GRADIENT_ALT  dp=1.5
            circles = cv2.HoughCircles(self.houghThresh[y1:y2,x1:x2],method=cv2.HOUGH_GRADIENT,dp=dp,minDist=int(minDist),param1=int(param1),param2=int(param2),minRadius=int(minR),maxRadius=int(maxR))
            if circles is not None:
                if showAll:#cv2.putText(self.overlay, 'CIRCLE', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,50,30),3)#
                        for i in circles[0,:]: cv2.circle(self.overlay[y1:y2,x1:x2,:],(int(i[0]),int(i[1])),int(i[2]),(0,255,255),2)
                weld = np.around(np.average(circles, axis=1)).astype(int)
                welds = np.concatenate((welds,weld),axis=0)
        except: raise(Exception)#None
        try:
            if showAvg and welds.any():
                weldAvg = np.average(welds,axis=0,weights=welds.any(axis=1)).astype(int)
                cv2.circle(self.overlay[y1:y2,x1:x2,:],(weldAvg[0],weldAvg[1]),weldAvg[2],(100,255,100),2)
                cv2.circle(self.overlay[y1:y2,x1:x2,:],(weldAvg[0],weldAvg[1]),5,(100,255,100),-1)

            '''if status:           
                try:
######## weld r min 52
                    delta = math.sqrt((w_center[0]-weldAvg[0]-rx1)**2 + (w_center[1]-weldAvg[1]-ry1)**2)
                    if delta < 52:
                        cv2.putText(frame_down,'Weld is Good',(5,90),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0),2)
                    else:
                        cv2.putText(frame_down,'Check Weld',(5,90),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255),2)
                except:None'''
        except:
            raise(Exception)#None

        
        
        return welds
    
    def find_weld_target(self, h_line, v_line, v_dist, h_dist, wRmin, wRmax, weld_target):
        weld_target = np.concatenate((weld_target,[[0,0]]),axis=0)
        if len(weld_target) > 6: weld_target = np.delete(weld_target, [0,1,2], axis=0)
        if v_line is not None and h_line is not None:
            try:
                v_vec, v_norm = normal_vector(v_line)
                h_vec, h_norm = normal_vector(h_line)
                x_corner, y_corner = line_intersection(h_line,v_line)
                corner = int(x_corner),int(y_corner)
                cv2.circle(self.overlay, (corner[0],corner[1]), 5, (255,255,0), -1)
                weld_center = np.array([np.subtract(np.add(corner,tuple([x * v_dist for x in v_vec])),tuple([x * h_dist for x in h_vec]))]).astype(int)
                #weld_center = np.array([np.subtract(np.add(corner,tuple([x * h_dist for x in h_vec])),tuple([x * v_dist for x in v_vec]))]).astype(int)
                weld_target = np.concatenate((weld_target,weld_center),axis=0)
            except:
                raise(Exception)
        try:
            if weld_target.any():
                w_center = np.average(weld_target,axis=0,weights=weld_target.any(axis=1)).astype(int)
                cv2.circle(self.overlay, (int(w_center[0]),int(w_center[1])), int(wRmin), (255,0,255), 2) 
                if True: cv2.circle(self.overlay, (int(w_center[0]),int(w_center[1])), int(wRmax), (255,0,255), 2)
        except: raise(Exception)#Nonebreakpoint()
    
        return weld_target
                
            
            

            
        '''if values["-SHOW MORE-"]:
                cv2.putText(frame_down,'Corner: ('+str(corner_centroid[0])+','+str(corner_centroid[1])+')',(5,990),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                cv2.putText(frame_down,'Weld Bound:('+str(int(w_center[0]))+','+str(int(w_center[1]))+')',(5,910),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                cv2.putText(frame_down,'Weld Radius:['+str(int(values["-WELD R MIN-"]))+','+str(int(values["-WELD R MAX-"]))+']',(5,950),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)'''

            
        
        
    def old_find_edges(self, v_x1, v_x2, v_y1, v_y2, h_x1, h_x2, h_y1, h_y2, acc_thresh, v_min, v_max, h_min, h_max):
        v_lines = cv2.HoughLinesP(self.v_edges[v_y1:v_y2,v_x1:v_x2], 1, np.pi/180, 50, acc_thresh, values["-minLineLength-"], values["-maxLineGap-"])#Vertical Edge
        h_lines = cv2.HoughLinesP(self.edges[h_y1:h_y2,h_x1:h_x2], 1, np.pi/180, 50, acc_thresh, values["-minLineLength-"], values["-maxLineGap-"])#Horizontal Edge
        
        if h_lines is not None:#Finds horizontal lines
            if True:
                for l in h_lines:
                    cv2.line(self.overlay[h_y1:h_y2,h_x1:h_x2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)#Draw Edge Line
            '''h2 = [l[0][1] for l in h_lines]#Extract Y values of each line
            l= h_lines[h2.index(min(h2))][0]#line with Y closest to weld
            P1,P2 = np.asarray((l[0]+h_x1,l[1]+h_y1)),np.asarray((l[2]+h_x1,l[3]+h_y1))#P1 P2 on edge line
            h_line = (P1,P2)
            if values["-SHOW MORE-"]:#Shows line
                cv2.line(frame[h_y1:h_y2,h_x1:h_x2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)#Draw Edge Line'''
                    
        if v_lines is not None:#Finds vertical lines
            v2 = [l[0][0] for l in v_lines]#Extract X values of each line
            l = v_lines[v2.index(min(v2))][0]#d_x_avg first line (most confident)            
            P1,P2 = np.asarray((l[0]+v_x1,l[1]+v_y1)),np.asarray((l[2]+v_x1,l[3]+v_y1))#P1 P2 on edge
            v_line = (P1,P2)
            v_line = ([x,y],[w,z])
            if values["-SHOW MORE-"]:#Shows lines
                cv2.line(frame[v_y1:v_y2,v_x1:v_x2], (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)#Draw line
           

    def sighlight(self, x, y, W, H, w, h, G):#creates rectangle w*h around point (x,y) and highlights threshold parts of image
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

            cv2.putText(self.overlay, 'SUM:{}'.format(self.gap_area), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),3)
            #cv2.putText(self.overlay, 'test', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),1)
            cv2.rectangle(self.overlay, (x1,y1), (x2,y2), (0,255,0),1)
        except:
            raise(Exception)
        
    def old_region(self):
        cv2.putText(self.img, 'Click on Region of Interest, Space to confirm',(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,100,100),3)
        cv2.namedWindow('ROI',cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('ROI', 30, 30)
        #cv2.resizeWindow('ROI', self.width//3, self.height//3)
        #cv2.resize(self.img, (self.width//3, self.height//3))
        x, y, _, _ = cv2.selectROI('ROI', cv2.pyrDown(self.img))
        print('x:{}, y:{}'.format(x*2,y*2))
        cv2.destroyAllWindows()
        return {'x': x*2, 'y': y*2}



class Camera:
    def __init__(self, cam_id, Gain, Exposure, FPS):
        self.system = PySpin.System.GetInstance()
        self.list = self.system.GetCameras()
        self.cam = self.list.GetBySerial(cam_id)
        self.cam.Init()
        
        self.nodemap = self.cam.GetNodeMap()
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

        #SET EXPOSURE, GAIN
        #breakpoint()
        PySpin.CEnumerationPtr(self.cam.GetTLStreamNodeMap().GetNode('StreamBufferHandlingMode')).SetIntValue(PySpin.CEnumerationPtr(self.cam.GetTLStreamNodeMap().GetNode('StreamBufferHandlingMode')).GetEntryByName('NewestOnly').GetValue())
        
        self.ID = PySpin.CStringPtr(self.nodemap.GetNode('DeviceID')).GetValue()
        print('Starting camera {}  '.format(cam_id),end='')      
#Turn off ExposureAuto, FramerateAuto, GainAuto
        PySpin.CEnumerationPtr(self.nodemap.GetNode('ExposureAuto')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('ExposureAuto')).GetEntryByName('Off').GetValue())
        PySpin.CEnumerationPtr(self.nodemap.GetNode('GainAuto')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('GainAuto')).GetEntryByName('Off').GetValue())
        PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode')).SetIntValue(PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode')).GetEntryByName('Continuous').GetValue())
        PySpin.CBooleanPtr(self.nodemap.GetNode('AcquisitionFrameRateEnable')).SetValue(True)
        #self.FPS = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))

        PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime')).SetValue(Exposure)
        PySpin.CFloatPtr(self.nodemap.GetNode('Gain')).SetValue(Gain)
        PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate')).SetValue(FPS)
        #breakpoint()
        print('\nExposure = {:.2f}\nGain = {:.2f}\nFPS = {:.2f}'.format(PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('Gain')).GetValue(),PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate')).GetValue()),end='\n')
        self.cam.BeginAcquisition()
    
    def get_image(self):
        return self.cam.GetNextImage().GetData().reshape(self.height,self.width,1)

    def exit(self, *args):
        self.list.Clear()
        self.cam.EndAcquisition()
        del self.cam
        self.system.ReleaseInstance()


def normal_vector(line):#Returns the vector of a line and its normal vector
    vector = (line[2]-line[0],line[3]-line[1])
    normal = (-vector[1],-vector[0])
    mag = math.sqrt(vector[0]**2 + vector[1]**2)
    unit_vector = (vector[0]/mag,vector[1]/mag)
    unit_normal = (-unit_vector[1],-unit_vector[0])
    return unit_vector, unit_normal

def line_intersection(line1, line2):#Returns the intersection of two lines
    xdiff = [line1[0] - line1[2], line2[0] - line2[2]]
    ydiff = [line1[1] - line1[3], line2[1] - line2[3]]
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        return 0, 0
    else:
        d = (det([line1[0],line1[1]],[line1[2],line1[3]]),det([line2[0],line2[1]],[line2[2],line2[3]]))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

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
        if os.path.isfile('{}save.txt'.format(cams['cam0'])):   settings = pickle.load(open('{}save.txt'.format(cams['cam0']),'rb'))
    if N_cams == 2 or not os.path.isfile('{}save.txt'.format(cams['cam0'])):
        print('No settings found, starting from scratch...')
        settings = {
            'Exposure':15000,
            'Gain':20,
            'FPS':10
            }
        options = list(cams.values())
        if N_cams == 2: options.append('Both')
        config_layout = [
            [sg.Text('Choose a camera:',    justification='left'),],
            [sg.Listbox(values=options,size=(15, 3), key='CAM ID'),sg.Text('',size=(2,3))],#*(len(cams)==2)              
             #sg.Spin([i for i in range(35)], settings['FPS'], key='FPS', size=(10,10))],

            [sg.Frame('Camera Settings',[
            [sg.Text('FPS',size=(9,1)),sg.InputText(settings['FPS'],size=(9,1),key='FPS')],
            [sg.Text('Exposure',size=(9,1)),sg.InputText(settings['Exposure'],size=(9,1),key='Exposure')],
            [sg.Text('Gain',size=(9,1)),sg.InputText(settings['Gain'],size=(9,1),key='Gain')]])],
             #sg.Spin([i for i in range(35)], settings['FPS'], key='FPS', size=(10,10))
             #sg.Spin([i for i in range(35)], settings['FPS'], key='FPS', size=(10,10))],
            [sg.Submit(), sg.Cancel()]
        ]
        window = sg.Window('Window Title', config_layout)    
        _, v = window.read()
        if v['CAM ID'] == []:
            if N_cams == 1: v['CAM ID'] = cams['cam0']
            if N_cams == 2: v['CAM ID'] = 'Both'
        else:
            v['CAM ID'] = v['CAM ID'][0]
        #settings.update(pickle.load(open('GUIsave.txt','rb')))
        settings.update({'re_setting':False})
        settings.update(v)
        window.close()

    #settings.update({'Exposure':50000,'Gain':20,'FPS':15})    
    settings.update(cams)
    cam_list.Clear()
    return settings

def load_settings(cam_id):
    if os.path.isfile('{}save.txt'.format(cam_id)): return pickle.load(open('{}save.txt'.format(cam_id),'rb'))
    else                                          : return {'CAM ID':cam_id}
    
    
def main():
    i = 0
    values = {}
    settings = startup()
    if settings['CAM ID'] == 'Both':
        win0 = Instance(load_settings(settings['cam0']))
        win1 = Instance(load_settings(settings['cam1']))
        wins = [win0, win1]                        
        for win in wins:
#TO DO, IF 2 CAMS EXPOSURE GAIN AND FPS WILL BE SAME BETWEEN THEM
            win.cam = Camera(win.config['CAM ID'],int(settings['Gain']),int(settings['Exposure']),int(settings['FPS']))
    else:
        win = Instance(settings)
        wins = [win]
        win.cam = Camera(settings['CAM ID'],int(settings['Gain']),int(settings['Exposure']),int(settings['FPS']))
        
    while True:
        for win in wins:
            win.loop(i)
        i+=1
        if i > 30: i =0

if __name__ == '__main__':
    main()
