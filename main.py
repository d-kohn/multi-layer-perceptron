#import numpy as np
from mnist import MNIST
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
INPUT_SIZE = len(images_training[0])

#index = random.randrange(0, len(images_training))  # choose an index ;-)
# print(mndata.display(images[index]))
training_target_counts = [0 for i in range(OUTPUT_N)]
print("Normalizing Training Data...")
for i in range(len(images_training)/4):
    if i % 10000 == 0:
        print(str(i) + "/60000 complete")
    
    for j in range(len(images_training[i])):
        images_training[i][j] /= 255
    images_training[i].insert(0, BIAS)
    training_target_counts[labels_training[i]] += 1

test_target_counts = [0 for i in range(OUTPUT_N)]
print("Normalizing Test Data...")
for i in range(len(images_test)/4):
    for j in range(len(images_test[i])):
        images_test[i][j] /= 255
    images_test[i].insert(0, BIAS)
    test_target_counts[labels_test[i]] += 1
#print(images_training[index])
# print(labels[index])

hidden_layer = []
hidden_layer_output = []
for index in range(HIDDEN_N+1): 
    hidden_layer.append(perceptron.perceptron(INPUT_SIZE, BIAS, ETA, MOMENTUM))
    hidden_layer_output.append(0)

output_layer = []
output_layer_output = []
for index in range(OUTPUT_N):   
    output_layer.append(perceptron.perceptron(HIDDEN_N, BIAS, ETA, MOMENTUM))
    output_layer_output.append(0)


targets = [0.1 for i in range(OUTPUT_N)]
for epoch in range(50):
    training_confusion_matrix = [[0 for i in range(OUTPUT_N+1)] for j in range(OUTPUT_N)]
    testing_confusion_matrix = [[0 for i in range(OUTPUT_N+1)] for j in range(OUTPUT_N)]

    print("Begin epoch ", epoch)
    for index in range(len(images_training)/4):
        if index % TRAINING_UPDATE_FREQ == 0:
            print("Image index: ", index)

        for id in range(HIDDEN_N):
            hidden_layer_output[id] = hidden_layer[id].activation(images_training[index])
        for id in range(OUTPUT_N):
            output_layer_output[id] = output_layer[id].activation(hidden_layer_output)
            if output_layer_output[id] >= 0.9:
                training_confusion_matrix[labels_training[index]][id] += 1

        targets[labels_training[index]] = 0.9
        output_layer_error = calc_output_layer_error(output_layer_output, targets)
        targets[labels_training[index]] = 0.1
        hidden_layer_error = calc_hidden_layer_error(hidden_layer_output, output_layer_error)

        for id in range(OUTPUT_N):
            output_layer[id].update_weights(output_layer_error[id], hidden_layer_output)
        for id in range(HIDDEN_N):
            hidden_layer[id].update_weights(hidden_layer_error[id], images_training[index])
        
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
        for id in range(HIDDEN_N):
            hidden_layer_output[id] = hidden_layer[id].activation(images_test[index])
        for id in range(OUTPUT_N):
            output_layer_output[id] = output_layer[id].activation(hidden_layer_output)
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

