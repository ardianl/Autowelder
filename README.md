# Autowelder
This program was developed to be used with FLIR BFS-U3-50S5C cameras. The program calculates the size of a gap in a selected region and displays pass/fail depending on how that size compares to the size limits set. To increase stability a fixture point feature is included which tracks a color/feature and adjusts coordinates as that point moves. The program initializes and connects to the camera to receive images, and uses OpenCV to process the images. PySimpleGUI is used to display and interact with the image processing settings. The program is used to locate the center of a weld and calculate the distance from the true position.

#Percussion Welder Program
This program is deprecated but remains for reference purposes. Requires GUIsave.txt to run. Code is unorganized and was refactored during Autowelder development. The program connects to the camera and receives images which are processed using OpenCV and displayed using PySimpleGUI. The program is used to capture images after a welding operation. The images are processed to locate the centerpoint of a weld and compares it against the calculated weld target point to judge the weld as good or bad. The camera capture drops frames and lags, code is unoptimized and runs slow. Issues with consistency lead to development of Autowelder.
