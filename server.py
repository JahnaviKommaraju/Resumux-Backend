from flask import Flask,render_template,request,jsonify
import os
import final
import time
from flask import json
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__,template_folder='template',static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/')
def home():
   return render_template('index.html')


@app.route('/upload',methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file'] 
      print(file)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

      op=final.final_function(os.path.join(app.config['UPLOAD_FOLDER']+"\\"+file.filename))
      response =jsonify(op)
      response.headers.add("Access-Control-Allow-Origin", "*")


      return response



if __name__ == '__main__':
   app.run(debug=True,port=80)
