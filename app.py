from flask import Flask, render_template, request, send_file
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
model = tf.keras.models.load_model("static/model.h5")
app = Flask(__name__)


def pred_image(arr, threshold=0.5):
    res = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
    for i, j in enumerate(arr):
        final_image = j[:, :, 0]
        final_image = prob_to_image(final_image, threshold=threshold)
        res[i] = final_image
    return res


def prob_to_image(arr, threshold=0.5):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i, j] > threshold:
                arr[i, j] = 1
            else:
                arr[i, j] = 0
    return arr


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("uploaded_file.png")
        z = plt.imread("uploaded_file.png")
        z = z*255
        z = np.sum(z, 2)
        z = np.pad(z, (13, 14))
        z = np.stack([z, np.full((128, 128), 506)], axis=2)
        z = model.predict(np.expand_dims(z, 0), verbose=0)
        z = z[:, 13:114, 13:114]
        plt.imsave("send.png", 255*pred_image(z, threshold=0.5)[0])

    return send_file("send.png", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
