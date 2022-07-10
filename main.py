import numpy as np
from mnist import MNIST
import random
import perceptron

BIAS = 1
HIDDEN_N = 20
OUTPUT_N = 10
ETA = 0.1

mndata = MNIST('samples')
images_training, labels_training = mndata.load_training()
images_test, labels_test = mndata.load_testing()
target_counts = [0 for i in range(OUTPUT_N)]
INPUT_SIZE = len(images_training[0])

#index = random.randrange(0, len(images_training))  # choose an index ;-)
# print(mndata.display(images[index]))

for i in range(len(images_training)):
    for j in range(len(images_training[i])):
        images_training[i][j] /= 255
    images_training[i].insert(0, BIAS)

for i in range(len(images_test)):
    for j in range(len(images_test[i])):
        images_test[i][j] /= 255
    images_test[i].insert(0, BIAS)
    target_counts[labels_test[i]] += 1
#print(images_training[index])
# print(labels[index])

net_layers = [perceptron.perceptron_output(INPUT_SIZE, BIAS) for i in ]
layer = []
# for output_target in range(OUTPUT_N): 
    
#     layer.append(perceptron.perceptron_output(INPUT_SIZE, BIAS, output_target))

# for index in range(HIDDEN_N): 
#     hidden_layer.append(perceptron.perceptron_output(INPUT_SIZE, BIAS, output_target))


index = 0
outputs = [0 for i in range(OUTPUT_N)]
targets = [0 for c in range(OUTPUT_N)]
for epoch in range(100):
    confusion_matrix = [[0 for i in range(OUTPUT_N+1)] for j in range(OUTPUT_N)]
    print("Begin epoch ", epoch)
    for index in range(len(images_training)):
        if index % 10000 == 0:
            print("Image index: ", index)
        targets[labels_training[index]] = 1
        for id in range(OUTPUT_N):
            outputs[id] = output_layer[id].train(images_training[index], targets[id], ETA)
        targets[labels_training[index]] = 0

    for index in range(len(images_test)):
        if index % 1000 == 0:
            print("Test index: ", index)
        for id in range(OUTPUT_N):
            outputs[id] = output_layer[id].test(images_test[index])
            if outputs[id] == 1:
                confusion_matrix[labels_test[index]][id] += 1
    f = open("data.csv", "a")
    f.write(str(epoch))
    for i in range(OUTPUT_N):
        f.write("\n")
        f.write(str(target_counts[i]))
        for j in range(OUTPUT_N):
            f.write(",")
            f.write(str(confusion_matrix[i][j]))
    f.write("\n")              
    f.close()

