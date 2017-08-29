import pylab
caffe_root = '/Users/HZzone/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
import os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
# run scripts from caffe root

from caffe import layers as L, params as P
import numpy as np


# define network and return the protobuf of the net
def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


# write to .prototxt file
with open('./lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('../mnist_train_lmdb', 64)))

with open('./lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('../mnist_test_lmdb', 100)))

def train():
    caffe.set_mode_cpu()

    ### load the solver and create train and test nets
    solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.SGDSolver('./lenet_auto_solver.prototxt')
    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    niter = 200
    test_interval = 25
    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / test_interval)))
    output = np.zeros((niter, 8, 10))

    # the main solver loop
    for it in range(200):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['score'].data[:8]

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
            test_acc[it // test_interval] = correct / 1e4

if __name__ == "__main__":
    train()
