from flask import Flask, request, Response, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()