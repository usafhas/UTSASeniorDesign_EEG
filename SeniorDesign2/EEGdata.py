from flask import Flask, render_template, request
import sqlite3 as sql
from datetime import datetime
import random
import time

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/viewdata')
def view_data():
	return render_template('data.html')

@app.route('/addrec',methods = ['POST','GET'])
def addrec():
	#if request.method == 'POST':
		try:
			out = random.randrange(0,4)
			#print out
			if(out == 0):
				emot = "Happy"
			elif(out == 1):
				emot = "Neutral"
			else:
				emot = "Calm"
			#print emot
			date = datetime.now()
			#print date
			with sql.connect("EEG_data.db") as con:
				cur = con.cursor()

				cur.execute("INSERT INTO dataset (Output,Emotion,Date) VALUES (?,?,?)",(out,emot,date))

				con.commit()
				msg = "Record successfully added"
		except:
			con.rollback()
			msg = "error in insert operation"

		finally:
			return render_template("result.html", out = out, emot = emot, date = date, msg = msg)
			con.close()

@app.route('/list')
def list():
	con = sql.connect("EEG_data.db")
	con.row_factory = sql.Row
	
	cur = con.cursor()
	cur.execute("select * from dataset")

	rows = cur.fetchall();
	return render_template("list.html",rows = rows)


if __name__ == '__main__':
	"""conn = sql.connect('EEG_data.db')
	print "Opened database successfully";
	conn.execute('CREATE TABLE dataset (Output SMALLINT, Emotion TEXT, Date TIMESTAMP)')
	print "Table created successfully";
	conn.close()"""
	app.run(debug = True)
"""
app.run( 
host="0.0.0.0",
port=int("80")
)
"""