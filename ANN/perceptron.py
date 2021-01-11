#!/usr/bin/env python
# coding: utf-8


OR_train_data = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
AND_train_data = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
NOR_train_data = [[0,0,1], [0,1,0], [1,0,0], [1,1,0]]
NAND_train_data = [[0,0,1], [0,1,1], [1,0,1], [1,1,0]]

w1 = 0.5
w2 = 0.2
b = -2
eta = 0.7

for epoch in range(1, 10):
    # randomly select a datapoint
    print("Epoch: " + str(epoch))
    for data in NAND_train_data:
        x1 = data[0]
        x2 = data[1]
        label = data[2]
        # prediction
        pred = w1*x1 + w2*x2 - b

        # if mistake in prediction; correct
        if label == 1 and pred <= 0:
            w1 = w1 + eta*x1
            w2 = w2 + eta*x2
            b = b - eta

        if label == 0 and pred > 0:
            w1 = w1 - eta*x1
            w2 = w2 - eta*x2
            b = b + eta
        
        print(label, 1 if pred>0 else 0, w1, w2, b)

    

