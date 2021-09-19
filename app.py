from flask import Flask, render_template, jsonify, request
import classification_model_tfidf

app = Flask(__name__)


@app.route("/MLService/association", methods=['GET', 'POST'])
def associationRuleMine():
    pass


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
