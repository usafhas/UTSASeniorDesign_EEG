# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:15:43 2016

@author: MichaelAAmaya
"""

from Tkinter import *
import threading

class App (object):
    """ Initialize App window, frames, and labels for basic architecture """
    def __init__(self):
		self.root=Tk()
		self.root.title("Test in progress")
		#self.root.resizable(width=False, height=False)
		#self.root.geometry("500x200")
		self.root.minsize(width=500, height=200)
		#self.root.pack_propagate(0) 

		self.emotion=StringVar()
		self.Count=StringVar()
  
		""" Emotion display counter Labels """
		self.Happy = StringVar()
		self.Angry = StringVar()
		self.Sad = StringVar()
		self.Neutral = StringVar()
		
		""" Emotion Counters """
		self.Happy_Count = 0
		self.Angry_Count = 0
		self.Sad_Count = 0
		self.Neutral_Count = 0
		
		
		""" Pack Frame setup """
		self.topframe = Frame(self.root, bg = 'white', width = 400, height = 100) # 360 X 80
		self.topframe.pack_propagate(0)
		self.topframe.pack(side = 'top', fill = 'x', anchor = 'nw')
		self.frame = Frame(self.root, bg = 'white', width = 360, height = 80)
		self.frame.pack_propagate(0)
		self.frame.pack(fill = 'x')#, anchor = 'w')
		self.buttonframe = Frame(self.root, bg = 'white')
		self.buttonframe.pack(fill = 'x')
		""" -------------------------------- """

		self.Title_message=Message(self.topframe, text = "EEG Output")#, textvariable=self.variable01) # textvariable overrides text
		self.Title_message.config(bg='red', font=('times', 18, 'italic'))
		
		self.subTitle_message=Message(self.frame, text = "Totals", width = 60)#, textvariable=self.variable01) # textvariable overrides text
		self.subTitle_message.config(bg='lightgreen', font=('times', 14, 'italic'))
		
		""" Label Setup """
		self.Emotion_label = Label(self.topframe, bg = 'white', text="Output Emotion: ", font=("Helvetica", 16), height = 2)
		self.Emotion_output=Label(self.topframe, bg = 'white', textvariable=self.emotion, font=("Helvetica", 16), height = 2)
		
		self.Happy_label=Label(self.frame, bg = 'white', text="Happy:", font=("Helvetica", 14))
		self.Happy_output=Label(self.frame, bg = 'white', textvariable=self.Happy, font=("Helvetica", 14), width = 4)
		
		self.Normal_label=Label(self.frame, bg = 'white', text = "Normal:", font=("Helvetica", 14))
		self.Normal_output=Label(self.frame, bg = 'white', textvariable=self.Neutral, font=("Helvetica", 14), width = 4)

		self.Angry_label=Label(self.frame, bg = 'white', text = "Angry:", font=("Helvetica", 14))
		self.Angry_output=Label(self.frame, bg = 'white', textvariable=self.Angry, font=("Helvetica", 14), width = 4)

		self.Sad_label=Label(self.frame, text = "Sad:", font=("Helvetica", 16))
		self.Sad_output=Label(self.frame, textvariable=self.Sad, font=("Helvetica", 16))
		
		self.Count_label=Label(self.buttonframe, bg = 'white', text = "Iteration:", font=("Helvetica", 14))
		self.Count_output=Label(self.buttonframe, bg = 'white', textvariable=self.Count, font=("Helvetica", 14), width = 4)

		
		return
        
    def grid(self):
		""" pack layout manager """
		self.Title_message.pack(fill='x', side = 'top')
		#self.Emotion_label.pack(fill='x', side = 'left', anchor = 'nw')
		#self.Emotion_output.pack(fill='x', side = 'left', anchor = 'nw')
		self.Emotion_output.pack(fill='x', anchor = 'center')
		
		self.subTitle_message.pack(fill='x', side = 'top', pady = 5)#, pady = 20)
		self.Happy_label.pack(fill='x', side = 'left', anchor = 'nw')
		self.Happy_output.pack_propagate(0) # keeps label from minimizing then expanding
		self.Happy_output.pack(fill='x', side = 'left', anchor = 'nw')
		self.Normal_label.pack(fill='x', side = 'left', anchor = 'nw')
		self.Normal_output.pack_propagate(0)
		self.Normal_output.pack(fill='x', side = 'left', anchor = 'nw')
		self.Angry_label.pack(fill='x', side = 'left', anchor = 'nw')
		self.Angry_output.pack_propagate(0)
		self.Angry_output.pack(fill='x', side = 'left', anchor = 'nw')
		
		self.Count_label.pack(fill='x', side = 'left', anchor = 'nw')
		self.Count_output.pack_propagate(0)
		self.Count_output.pack(fill='x', side = 'left', anchor = 'nw')
		#self.Sad_message.pack(fill=X, side = LEFT)
		#self.Sad_label.pack(fill='x', side = 'left')
		#self.Sad_output.pack(fill='x', side = 'left')
        #return
        
    def update_label(self, Output, iteration):
		self.Count.set(str(iteration))
        
		if(Output == 0):
			self.emotion.set("Normal" + " (" + str(Output) + ")")
			self.Neutral_Count = self.Neutral_Count+1;
			self.Neutral.set(str(self.Neutral_Count))
		elif(Output == 1):
			self.emotion.set("Happy" + " (" + str(Output) + ")")
			self.Happy_Count = self.Happy_Count+1;
			self.Happy.set(str(self.Happy_Count))
		else:
			self.emotion.set("Angry" + " (" + str(Output) + ")")
			self.Angry_Count = self.Angry_Count+1
			self.Angry.set(str(self.Angry_Count))
		
   
		self.root.update_idletasks()
		self.root.update()
		return
        
    """ Must create button separately to be able to close out of window """
    def button(self):
        self.quit_button = Button(self.buttonframe, text='Stop', width=10, height=1,command=self.root.destroy)
        self.quit_button.pack(side = 'right', anchor = 'se')
        return
        
    """ Sets up window with labels and variales """    
    def setup(self):
        self.grid()
        self.button()

if __name__=='__main__':
	#App(100).run()
	App1 = App()
	App1.setup()
	i = 0
	j = 0
	while(1): 
		i = 900000
		#App.update_label(App1, j % 3, j)
		App1.update_label(j % 3, j)
		while(i >= 0):
			i = i-1
		j = j+1
		print("i: %s j: %s\n" % (i,j)) 
		raw_input('--> ')
        
        