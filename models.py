import json
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from bert4keras.models import build_transformer_model


def get_model(num_labels, config_path, checkpoint_path, concat_last=1, pooling='cls_out', trainable=True):
    assert pooling in ['cls_out', 'avg_pooling']
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )

    # set bert layers untrainable
    for l in model.layers:
        l.trainable = trainable
    
    input_file = open(config_path)
    json_array = json.load(input_file)
    bert_layers = json_array['num_hidden_layers']

    output_layers = []
    for i in range(concat_last):
        i += 1
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - i)
        if pooling == 'cls_out':
            output = keras.layers.Lambda(lambda sequence_out: sequence_out[:, 0, :])(model.get_layer(output_layer).output)
        elif pooling == 'avg_pooling':
            output = keras.layers.GlobalAveragePooling1D()(model.get_layer(output_layer).output)
        output_layers.append(output)
    if len(output_layers) < 2:
        output = output_layers[0]
    else:
        output = keras.layers.concatenate(output_layers)
    output = keras.layers.Dense(output.shape[1])(output)
    output = keras.layers.BatchNormalization()(output)
    output = keras.layers.Activation(tf.nn.relu)(output)
    output = keras.layers.Dropout(0.1)(output)
    output = keras.layers.Dense(num_labels)(output)
    model = keras.models.Model(model.input, output)
    model.summary()
    return model