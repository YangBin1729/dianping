import json

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired

from models import predict
from config import Config

app = Flask(__name__)
config = Config()
app.config.from_object(Config)



class InputForm(FlaskForm):
    content = TextAreaField('输入想要分析的内容', validators=[DataRequired()],
                            render_kw={"rows": 10})
    submit = SubmitField('解析')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    results = None
    form = InputForm()
    if form.validate_on_submit():
        content = form.content.data
        results = predict(config, content)
        results = json.dumps(results)
        print(results)
    return render_template('classify.html', form=form, results=results)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=True)