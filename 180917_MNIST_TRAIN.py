# python script for an self-made neuronal network
# code for a 3-layer neural network, and code for learning the MNIST dataset

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# library for suppressing exponential within decimal numbers
import decimal
# helper to load data from PNG image files
import imageio
import scipy.misc
# glob helps select multiple files using patterns
import glob
# library to resize image
import cv2



# neuronal network class definition
class neuralNetwork:

    # initialise the neuronal network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih (weight input layer to hidden layer) and who (hidden layer to output layer)
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # OPTION 1: zufallsbasierte Initialisierung der Gewichte:
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        # OPTION 2: Stichproben aus einer Normalverteilung mit dem Mittelwert 0 und einer Standardabweichung,
        # die auf der Anzahl der Verknüpfungen zu einem Knoten basiert (1/Wurzel(Anzahl der eingehenden Verknüpfungen)
        # pow -> Anzahl der Knoten hoch -0.5 -> Wurzel aus Anzahl der Knoten
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, - 0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, - 0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        self.counter = 0

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array (.T returns the transposed matrix)
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # = Punktprodukt von zwei Matrizen
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and the output layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and the hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        # return self.wih
        # return self.wih
        # print(self.wih)

        self.counter = self.counter + 1
        #print("Trainingslauf #", self.counter)
        pass

    #query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array (https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass


# number of input, hidden, output nodes
input_nodes = 784
hidden_nodes = 200 # Best: 200
output_nodes = 10

# learning rate
learning_rate = 0.2

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# print("show weights from input to hidden layer: \n", n.wih, "\n")
# print("single run through neuronal network, final outputs: \n", n.query([1.0, 0.5, -1.5]),"\n")
# print("modified weights: \n", n.train([1.0, 0.5, -1.5], [0.5, 1.5, -3.5]))

# load the mnist training data CSV file into a list
training_data_file = open("//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# TRAIN the neuronal network

# epochs is the number of times the training data set is used for training
epochs = 1 # Best: 2-5
train_counter = 0

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        train_counter = train_counter + 1
        # print("Epoche:", epochs, "| Image:", train_counter)
        pass
    pass


# load the mnist test data CSV file into a list
test_data_file = open("//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# TEST the neuronal network - OPTION 1 "Single Image"
# get the first test record
all_values = test_data_list[0].split(',')
# print the label
# print(all_values[0])

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# matplotlib.pyplot.show()

#avoid to print float number with exponentials
numpy.set_printoptions(suppress=True)
#print output matrix
# print(n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
#matplotlib.pyplot.show()


# TEST the neuronal network - OPTION 2 "all Images"
# scorecard for how well the network performas, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # networks answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)

# TEST the neuronal network - OPTION 3 "My own RASPI-CAMERA Image"
# our own image test data set
our_own_dataset = []

image_file_name_target = '//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/custom/testdata/2828_my_ownd_5.png'

for image_file_name in glob.glob('//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/custom/2828_my_own_origd_5.png'):
    print ("loading ... ", image_file_name)
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])

    # resize image
    from PIL import Image

    img = Image.open(image_file_name)
    width, height = img.size
    ratio = numpy.floor(height / width)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img.save(image_file_name_target, format='PNG')

    # load image data from png files into an array
    # img_array = imageio.imread(image_file_name, as_gray=True)
    img_array = scipy.misc.imread(image_file_name_target, flatten=True)
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # modify raspi camera image - turn light gray areas to white (keep only the number)
    img_data[img_data < 0.5955] = 0

    # show image
    image_array = numpy.asfarray(img_data[0:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    print(numpy.max(img_data))

    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    print(record)
    our_own_dataset.append(record)

    # query the network
    outputs = n.query(img_data)
    print(outputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("network says ", label)
    pass

# Save model to CSV:
numpy.savetxt("//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/model/wih.csv", n.wih, delimiter=",")
numpy.savetxt("//vmware-host/Shared Folders/Workstation-Data/Service/Research/AI/Data/MNIST/model/who.csv", n.who, delimiter=",")