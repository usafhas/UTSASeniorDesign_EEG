# Template for Brainstate Graphical Output
# Senior Design
# UTSA
# Author: Garrett Hall
# 1/7/2017

from graphics import *
import random
import time

a = 0
win = GraphWin("My Circle", 300, 300)
while a < 10:
    a = a + 1
    z=random.randint(1,3);
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
    time.sleep(1)



#win.getMouse() # Pause to view result
#win.close()    # Close window when done

# main()