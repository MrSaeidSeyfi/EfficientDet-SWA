import tensorflow as tf

def load_efficientdet_model(model_dir):
    return tf.saved_model.load(model_dir)
