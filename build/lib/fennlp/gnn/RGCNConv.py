#! encoding="utf-8"
import tensorflow as tf


class RelationalGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim,
                 num_relations,
                 num_bases,
                 activation='linear',
                 kernel_initializer=None,
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 **kwargs):
        super(RelationalGraphConvolution, self).__init__(self, **kwargs)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

    def build(self, input_shape):
        self.in_features = input_shape[1]
        # V_b   [B,in_dim,out_dim]
        self.basis = self.add_weight(
            shape=(self.num_bases, self.in_features, self.output_dim),
            name='basis',
        )

        self.att = self.add_weight(
            shape=(self.num_relations, self.num_bases),
            name='kernel',
        )
        # W_0^l
        self.weight = self.add_weight(
            shape=(self.in_features, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name='kernel',
        )
        # bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                name='bias',
            )

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        # wr [R,in_dim*out_dim]
        wr = tf.linalg.matmul(self.att, tf.reshape(self.basis, (self.num_bases, -1)))

        if x_j is None:
            # wr [R*in_dm, out_dim]
            wr = tf.reshape(wr, (-1, self.output_dim))
            index = edge_type * self.in_features + edge_index_j
            output = tf.gather(wr, index)
        else:
            wr = tf.reshape(wr, (self.num_relations, -1, self.output_dim))
            wr = tf.gather(wr, tf.cast(edge_type, tf.int64))
            print(wr)
            output = tf.linalg.matmul(tf.expand_dims(x_j, 1), wr)
            output = tf.squeeze(output, -2)
        return output if edge_norm is None else output * tf.reshape(edge_norm, (-1, 1))

    def aggregate(self, outputs):
        return tf.reduce_mean(outputs)

    def call(self, x, edge_index, edge_type, edge_norm):
        """
        :param x: entity
        :param edge_index: edge_index
        :param edge_type: edge_type
        :param edge_norm:edge_norm
        :return:
        """
        output = self.message(x, edge_index, edge_type, edge_norm)
        # TODO
        aggr_out = self.aggregate(output)
        if x is None:
            output = aggr_out + self.weight
        else:
            output = aggr_out + tf.linalg.matmul(x, self.weight)
        if self.use_biase:
            output += self.bias
        return output
