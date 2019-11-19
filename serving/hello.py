from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return app.send_static_file('index.html')

def parsenb(file):
    code = ''
    parsednb = json.loads(file)
    for j in range(len(parsednb['cells'])):
        if parsednb['cells'][j]['cell_type'] == 'code':
            code = code + ''.join(parsednb['cells'][j]['source'])
    return code


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_name = file.filename
    notebook = str(file.read().decode('UTF-8'))
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(type(tokenizer))
    code = parsenb(notebook)
    code = np.array([code])
    print(code)
    code = tokenizer.texts_to_sequences(code)
    code = tf.keras.preprocessing.sequence.pad_sequences(code, padding='post', maxlen=2000)
    code = code.astype('float32')
    loaded = tf.saved_model.load("/tmp/lenet/3/")
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    pred = infer(tf.constant(code))['dense_4'].numpy()[0][0]
    
    if pred > .5:
        framework = 'TensorFlow'
    else:
        framework = 'PyTorch'
    returnstring = 'your top deep learning framework is:<br>' + framework 
    return returnstring


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
