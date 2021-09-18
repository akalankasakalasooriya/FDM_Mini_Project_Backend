from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route("/MLService/association", methods=['GET', 'POST'])
def associationRuleMine():
    pass


@app.route("/MLService/classification", methods=['GET', 'POST'])
def classify():
    print(request.json)
    return(jsonify({'hello': 'itworked'}))


if __name__ == "__main__":
    app.run()
