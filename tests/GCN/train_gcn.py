#! encoding:utf-8
import tensorflow as tf
from fennlp.datas import graphloader
from fennlp.models import GCN
from fennlp.optimizers import optim
from fennlp.metrics import Losess, Metric

_HIDDEN_DIM = 64
_NUM_CLASS = 7
_DROP_OUT_RATE = 0.5
_EPOCH = 100

loader = graphloader.GCNLoader()
features, adj, labels, idx_train, idx_val, idx_test = loader.load()

model = GCN.GCN2Layer(_HIDDEN_DIM, _NUM_CLASS, _DROP_OUT_RATE)

optimizer = tf.keras.optimizers.Adam(0.01)

crossentropy = Losess.MaskSparseCategoricalCrossentropy(from_logits=False)
accscore = Metric.SparseAccuracy()
f1score = Metric.SparseF1Score(average="macro")
# ---------------------------------------------------------
# For train
for epoch in range(_EPOCH):
    with tf.GradientTape() as tape:
        output = model(features, adj, training=True)
        predict = tf.gather(output, list(idx_train))
        label = tf.gather(labels, list(idx_train))
        loss = crossentropy(label, predict, use_mask=False)
        acc = accscore(label, predict)
        f1 = f1score(label, predict)
        print("Epoch {} | Loss {:.4f} | Acc {:.4f} | F1 {:.4f}".format(epoch, loss.numpy(), acc,f1))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# ------------------------------------------------------
# For Valid
output = model.predict(features, adj)
predict = tf.gather(output, list(idx_val))
label = tf.gather(labels, list(idx_val))
acc = accscore(label, predict)
f1 = f1score(label, predict)
loss = crossentropy(label, predict, use_mask=False)
print("Valid Loss {:.4f} | ACC {:.4f} | F1 {:.4f}".format(loss.numpy(), acc,f1))

# For test
output = model.predict(features, adj)
predict = tf.gather(output, list(idx_test))
label = tf.gather(labels, list(idx_test))
acc = accscore(label, predict)
f1 = f1score(label, predict)
loss = crossentropy(label, predict, use_mask=False)
print("Test Loss {:.4f} | ACC {:.4f} | F1 {:.4f}".format(loss.numpy(), acc,f1))
