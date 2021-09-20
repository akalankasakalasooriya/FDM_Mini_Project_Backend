from flask import Flask, render_template, jsonify, request
import classification_logistic_reg_model as lrm
import classification_linear_svm_model as lsm
import classification_random_forest_model as rfm
import classification_perceptron_model as pm
import classification_decision_tr_model as dtr

app = Flask(__name__)


@app.route("/MLService/association", methods=['GET', 'POST'])
def associationRuleMine():
    pass


@app.route("/MLService/classification", methods=['GET', 'POST'])
def classify():
    req_data = request.json
    if req_data['ALGO'] == 'Logistic Regression with OneVsRest and TF-IDF':
        predictions = lrm.get_predictions(
            req_data['PLOT'])
        print(predictions)
    elif req_data['ALGO'] == 'Linear SVC with OneVsRest and TF-IDF':
        predictions = lsm.get_predictions(
            req_data['PLOT'])
        print(predictions)
    elif req_data['ALGO'] == 'Random Forest Classifier with OneVsRest and TF-IDF':
        predictions = rfm.get_predictions(
            req_data['PLOT'])
        print(predictions)
    elif req_data['ALGO'] == 'Perceptron with OneVsRest and TF-IDF':
        predictions = pm.get_predictions(
            req_data['PLOT'])
        print(predictions)
    elif req_data['ALGO'] == 'Decision Tree Regressor with OneVsRest and TF-IDF':
        predictions = dtr.get_predictions(
            req_data['PLOT'])
        print(predictions)
    else:
        predictions = [['']]

    return(jsonify({'GENRES': ", ".join([x.capitalize() for x in predictions[0]])}))


if __name__ == "__main__":
    app.run()
