#! encoding="utf-8"
import tensorflow as tf
from fennlp.gnn import GCNConv


class GCN2Layer(tf.keras.Model):
    def __init__(self, hidden_dim, num_class, dropout_rate=0.5, **kwargs):
        super(GCN2Layer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.gc1 = GCNConv.GraphConvolution(hidden_dim)
        self.gc2 = GCNConv.GraphConvolution(num_class)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, adj, training=True):
        x = tf.nn.relu(self.gc1(inputs, adj))
        x = self.dropout(x, training=training)
        x = self.gc2(x, adj)
        return tf.math.softmax(x, -1)

    def predict(self, inputs, adj, training=False):
        return self(inputs, adj, training)
