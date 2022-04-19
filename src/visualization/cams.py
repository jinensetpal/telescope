from src.const import TARGET_SIZE, BASE_DIR, PENULTIMATE_LAYER
import matplotlib.pyplot as plt
from tensorflow import keras
import scipy as sp
import numpy as np
import os

def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label_index = np.argmax(predictions)
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]

    final_conv_layer = model.get_layer(PENULTIMATE_LAYER) 

    get_output = keras.backend.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = np.squeeze(conv_outputs)
    mat_for_mult = sp.ndimage.zoom(conv_outputs, (TARGET_SIZE[0] / conv_outputs.shape[0], TARGET_SIZE[1] / conv_outputs.shape[1], 1), order=1) # dim: 224 x 224 x 2048
    final_output = np.dot(mat_for_mult.reshape((TARGET_SIZE[0] * TARGET_SIZE[1], 64)), class_weights_winner).reshape(TARGET_SIZE[0], TARGET_SIZE[1]) # dim: 224 x 224

    return final_output, label_index

def main():
    from src.data.generator import get_generators
    train, val, test_X, test_y = get_generators(BASE_DIR)
    model = keras.models.load_model('models', 'cnn-real')

    fig = plt.figure(figsize=(14, 14),
                    facecolor='white')

    for idx in range(16):
            out, pred = get_class_activation_map(model, test_X[idx])

            fig.add_subplot(4, 4, idx + 1)
            buf = 'Predicted Class = ' + str(pred)
            plt.xlabel(buf)
            plt.imshow(test_X[idx], alpha=0.5)
            plt.imshow(out, cmap='jet', alpha=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join('visualizations', 'cams.png'))

if __name__ == "__main__":
    main()
