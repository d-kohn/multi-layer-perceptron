from operator import matmul
import numpy as np
from mnist import MNIST
import math
import perceptron

BIAS = 1
HIDDEN_N = 100
OUTPUT_N = 10
ETA = 0.1
MOMENTUM = 0.9
TRAINING_UPDATE_FREQ = 100

OUTPUT = 1
HIDDEN = 0

def calc_output_layer_error(output, target):
    error = []
    for i in range(OUTPUT_N):
        error.append(output[i]*(1 - output[i])*(target[i]-output[i]))
    return error

def calc_hidden_layer_error(output, output_layer_error):
    error = []
    for i in range(HIDDEN_N+1):
        error.append(output[i]*(1 - output[i])*hidden_layer[i].calc_error_sum(output_layer_error))
    return error


mndata = MNIST('samples')

print("Loading Training Data...")
images_training, labels_training = mndata.load_training()
print("Loading Test Data...")
images_test, labels_test = mndata.load_testing()
INPUT_SIZE = len(images_training[0])+1

#index = random.randrange(0, len(images_training))  # choose an index ;-)
# print(mndata.display(images[index]))
training_target_counts = [0 for i in range(OUTPUT_N)]
print("Normalizing Training Data...")
for i in range(len(images_training)):
    if i % 10000 == 0:
        print(str(i) + "/60000 complete")
    for j in range(len(images_training[i])):
        images_training[i][j] /= 255
    images_training[i].insert(0, BIAS)
    training_target_counts[labels_training[i]] += 1

test_target_counts = [0 for i in range(OUTPUT_N)]
print("Normalizing Test Data...")
for i in range(len(images_test)):
    for j in range(len(images_test[i])):
        images_test[i][j] /= 255
    images_test[i].insert(0, BIAS)
    test_target_counts[labels_test[i]] += 1

hidden_layer_weights = np.random.uniform(-0.05, 0.05, INPUT_SIZE*(HIDDEN_N+1)).reshape(INPUT_SIZE, HIDDEN_N+1)
hidden_delta_weights = np.empty(shape=(INPUT_SIZE,HIDDEN_N+1))
hidden_delta_weights.fill(0)
hidden_layer_output = np.empty(shape=(HIDDEN_N+1))
hidden_layer_errors = np.empty(shape=(HIDDEN_N+1))

output_layer_weights = np.random.uniform(-0.05, 0.05, (HIDDEN_N+1)*OUTPUT_N).reshape(HIDDEN_N+1, OUTPUT_N)
output_delta_weights = np.empty(shape=(HIDDEN_N+1,OUTPUT_N))
output_delta_weights.fill(0)
output_layer_output = np.empty(shape=(OUTPUT_N))
output_layer_error = np.empty(shape=(OUTPUT_N))

targets = [0.1 for i in range(OUTPUT_N)]
for epoch in range(50):
    training_confusion_matrix = [[0 for i in range(OUTPUT_N+1)] for j in range(OUTPUT_N)]
    testing_confusion_matrix = [[0 for i in range(OUTPUT_N+1)] for j in range(OUTPUT_N)]

    print("Begin epoch ", epoch)
    for index in range(len(images_training)):
        if index % TRAINING_UPDATE_FREQ == 0:
            print("Image index: ", index)
        z_h = matmul(images_training[index], hidden_layer_weights)
        hidden_layer_output = 1/(1+pow(math.e,-z_h))
        hidden_layer_output[HIDDEN_N] = BIAS
        z_o = matmul(hidden_layer_output, output_layer_weights)
        output_layer_output = 1/(1+pow(math.e,-z_o))

        targets[labels_training[index]] = 0.9
        for id in range(OUTPUT_N):
            if output_layer_output[id] >= 0.9:
                training_confusion_matrix[labels_training[index]][id] += 1

        output_layer_error = output_layer_output*(1 - output_layer_output)*(targets-output_layer_output)
        targets[labels_training[index]] = 0.1
        sums = np.empty(shape=(HIDDEN_N+1))
        for i in range(HIDDEN_N+1):
            sums[i] = np.sum(output_layer_weights[i]*output_layer_error)
        hidden_layer_error = hidden_layer_output*(1 - hidden_layer_output)*sums
        
        temp = np.tile(hidden_layer_output, (len(output_layer_error),1))
        temp = temp.transpose() * output_layer_error
        output_delta_weights = ETA*temp+MOMENTUM*output_delta_weights
        output_layer_weights = output_layer_weights + output_delta_weights
 
        temp = np.tile(images_training[index], (len(hidden_layer_error),1))
        temp = temp.transpose() * hidden_layer_error
        hidden_delta_weights = ETA*temp+MOMENTUM*hidden_delta_weights
        hidden_layer_weights = hidden_layer_weights + hidden_delta_weights
        
    f = open("training_data_n100.csv", "a")
    f.write(str(epoch))
    for i in range(OUTPUT_N):
        f.write("\n")
        f.write(str(training_target_counts[i]))
        for j in range(OUTPUT_N):
            f.write(",")
            f.write(str(training_confusion_matrix[i][j]))
        f.write(",")
        f.write(str(sum(training_confusion_matrix[i][0:OUTPUT_N])-training_confusion_matrix[i][i]))
    f.write("\n")              
    f.close()

    for index in range(len(images_test)):
        if index % 1000 == 0:
            print("Test index: ", index)
        z_h = matmul(images_test[index], hidden_layer_weights)
        hidden_layer_output = 1/(1+pow(math.e,-z_h))
        hidden_layer_output[HIDDEN_N] = BIAS
        z_o = matmul(hidden_layer_output, output_layer_weights)
        output_layer_output = 1/(1+pow(math.e,-z_o))
        targets[labels_test[index]] = 0.9
        for id in range(OUTPUT_N):
            if output_layer_output[id] >= 0.9:
                testing_confusion_matrix[labels_test[index]][id] += 1

    f = open("testing_data_n100.csv", "a")
    f.write(str(epoch))
    for i in range(OUTPUT_N):
        f.write("\n")
        f.write(str(test_target_counts[i]))
        for j in range(OUTPUT_N):
            f.write(",")
            f.write(str(testing_confusion_matrix[i][j]))
        f.write(",")
        f.write(str(sum(testing_confusion_matrix[i][0:OUTPUT_N])-testing_confusion_matrix[i][i]))
    f.write("\n")              
    f.close()

