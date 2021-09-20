from flask import Flask, render_template, jsonify, request
import classification_model_tfidf
import association_rules_finder

app = Flask(__name__)


@app.route("/MLService/association", methods=['GET', 'POST'])
def associationRuleMine():
    req_data = request.json
    result = []
    if req_data["ALGO"] == 'FPG':
        movi_list = req_data["MOVI_LIST"]
        result = association_rules_finder.get_rules_fp(movi_list)
    elif req_data["ALGO"]=='APR':
        movi_list = req_data["MOVI_LIST"]
        result = association_rules_finder.get_rules_ap(movi_list)
    return(jsonify({'MOVIES': str(result)}))


@app.route("/MLService/classification", methods=['GET', 'POST'])
def classify():
    req_data = request.json
    if req_data['ALGO'] == 'Logistic Regression with TF-IDF':
        predictions = classification_model_tfidf.get_predictions(
            req_data['PLOT'])
        print(predictions)
    elif req_data[''] == '':
        pass

    return(jsonify({'GENRES': ", ".join([x.capitalize() for x in predictions[0]])}))


if __name__ == "__main__":
    app.run()
