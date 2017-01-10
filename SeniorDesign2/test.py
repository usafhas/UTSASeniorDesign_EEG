from EEGdata import push
import random

out = random.randrange(0,4)
#print out
if(out == 0):
	emot = "Happy"
elif(out == 1):
	emot = "Neutral"
else:
	emot = "Calm"
#print emot

push(out, emot)


