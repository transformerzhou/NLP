import tensorflow as tf


class MaskSparseCategoricalCrossentropy():
    def __init__(self, from_logits, use_mask=False):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict, self.from_logits)
        if self.use_mask:
            input_mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
            cross_entropy = tf.reduce_sum(cross_entropy * input_mask) / (tf.reduce_sum(input_mask)+ 1e-6)   # mask loss
            return cross_entropy
        else:
            return tf.reduce_mean(cross_entropy)
