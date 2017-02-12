# Template for Brainstate Graphical Output
# Senior Design
# UTSA
# Author: Garrett Hall
# 1/7/2017

# edit: 1/8/2016
# added brain state labels to corresponding indicators
# provided comments
from __future__ import print_function
from graphics import *
import random
import time
import numpy as np
import pickle
import sys

# builds the window we will be drawing in
win = GraphWin("Brain State", 300, 300)

# these 3 labels tell us which brain state the system is in
label = Text(Point(150, 50), 'Angry')
label.setTextColor('red')
label.draw(win)

label = Text(Point(150, 150), 'Neutral')
label.setTextColor('blue')
label.draw(win)

label = Text(Point(150, 250), 'Calm')
label.setTextColor('green')
label.draw(win)

# this while loop demonstrates how inputs will be received. it counts to 10 and outputs a random integer 1,2, or 3
# we can let the variable 'z' be substituted for the output of 'codename_dutchess'
# the code uses the if/elif format to accommodate each brain state independently.
# this was done so residue images do not conflict with current brain states


a = 0
while a < 10:
    a = a + 1
#    z=random.randint(1,3);
    z = np.load('./result.npy')
    if z==1:
        c_red = Circle(Point(50, 50), 30)
        c_red.draw(win)
        c_red.setFill('red')
        c_red.setOutline('red')

        c_blue = Circle(Point(50, 150), 30)
        c_blue.draw(win)
        c_blue.setFill('white')
        c_blue.setOutline('white')

        c_green = Circle(Point(50, 250), 30)
        c_green.draw(win)
        c_green.setFill('white')
        c_green.setOutline('white')

    elif z==2:
        c_red = Circle(Point(50, 50), 30)
        c_red.draw(win)
        c_red.setFill('white')
        c_red.setOutline('white')

        c_blue = Circle(Point(50, 150), 30)
        c_blue.draw(win)
        c_blue.setFill('blue')
        c_blue.setOutline('blue')

        c_green = Circle(Point(50, 250), 30)
        c_green.draw(win)
        c_green.setFill('white')
        c_green.setOutline('white')

    elif z==3:
        c_red = Circle(Point(50, 50), 30)
        c_red.draw(win)
        c_red.setFill('white')
        c_red.setOutline('white')

        c_blue = Circle(Point(50, 150), 30)
        c_blue.draw(win)
        c_blue.setFill('white')
        c_blue.setOutline('white')

        c_green = Circle(Point(50, 250), 30)
        c_green.draw(win)
        c_green.setFill('green')
        c_green.setOutline('green')
    # the time.sleep command is used to pause the loop for 1 second
    time.sleep(5)

#win.getMouse() # Pause to view result
#win.close()    # Close window when done