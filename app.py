#data analysis and manipulation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

#backend
from flask import Flask, request, render_template 

#rendering plot
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
matplotlib.use('Agg')

#flask html destination
app = Flask(__name__, template_folder='templates')

#home page
@app.route("/")
def home():
    return render_template("home.html")

#graphing page from entered variables
@app.route('/plot', methods =["GET", "POST"])
def graph():
    if request.method == "POST":
        
        #user inputted csv data
        data = request.form.get("mycsv")

        #user inputted var elim
        eliminate = request.form.get("eliminate")
        eliminated = eliminate.split(',') #remove

        #target var
        target = request.form.get("target")
        eliminated.append(target) #add target
    
        #process plot img
        img = BytesIO()

        #read data
        df = pd.read_csv(data)
        X = df.drop(eliminated,axis=1)
        y = df[target]

        #train data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

        #lin regression
        lm = LinearRegression()
        lm.fit(X_train,y_train)

        #make predictions
        predictions = lm.predict(X_test)
        plt.scatter(y_test,predictions)
        plt.xlabel('Test Variable')
        plt.ylabel('Predictions')
        
        score = lm.score(X_test,y_test)
        score = score*100
        score = str(score)

        #process image
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("graph.html", plot_url = plot_url,accuracy=score)
    
if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)