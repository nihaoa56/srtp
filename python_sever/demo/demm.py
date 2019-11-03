from flask import Flask, request
import json
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/a',methods=["GET","POST"])
def test1():
    if request.method == 'POST':
        print("name:")
        print(request.form.get('name'))
        print("nameList:")
        print(request.form.getlist('name'))
        print("age:")
        print(request.form.get('age', default='-1'))
        return json.dumps(request.form)



if __name__ == '__main__':
    app.run(port=8838,debug=True)



