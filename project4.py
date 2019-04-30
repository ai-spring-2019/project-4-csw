"""
NAME: Colleen Wahl
DATE: ?
PROJECT:

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here
class NetNode():
    """ A node in the neural network. """
    
    def __init__(self, a = 'None', error = 'None'):
        
        self._a = a
        self.error = error
        self.parents = []
        self.p_weights = []
        self.children = []
        self.c_weights = []
    
    def set_a(self, a):
        self._a = a
        
    def get_a(self):
        return self._a
    

class NeuralNetwork():
    def __init__(self, node_num_lst, alpha = 0.1, decrease = False):
        
        self.alpha = alpha
        
        # determines whether alpha will get smaller as we go
        self.decrease = decrease
        
        # each item in the list is the number of nodes in that layer,
        # so node_num_lst[0] is the # of nodes in the first layer, etc.
        dummy = NetNode(1)
        self.in_nodes = [dummy]
        for _ in range(node_num_lst[0]):
            self.in_nodes.append(NetNode())
                         
        # contains nodes in hidden layers and the output layer
        self.other_nodes = []
        for i in range(1, len(node_num_lst)):
            this_layer = []
            for _ in range(node_num_lst[i]):
                new_node = NetNode()
                this_layer.append(new_node)
                if i == 1:
                    for p_node in self.in_nodes:
                        self.connect(p_node, new_node)
                else:
                    for p_node in self.other_nodes[i-2]:
                        self.connect(p_node, new_node)
            self.other_nodes.append(this_layer)
            
        self.out_nodes = self.other_nodes[-1]
                
        
    def connect(self, n1, n2, weight = 'None'):
        """ Connect nodes n1 and n2 with a path of the given weight.
            Assumes n1 is the parent node. """
        
        if weight == 'None':
            weight = random.random()

        n1.children.append(n2)
        n1.c_weights.append(weight)
        n2.parents.append(n1)
        n2.p_weights.append(weight)
            
    def predict_class(self):
        pass
    
    def forward_propagate(self, input_lst):
        """ Given a list of inputs, determine all nodes' activations. """
        
        for i in range(len(input_lst)):
            # add 1 to i to skip over the dummy node.
            # set activations for input nodes = to the input.
            self.in_nodes[i+1].set_a(input_lst[i])
            
        for layer in self.other_nodes:
            for node in layer:
                in_j = 0
                for t in range(len(node.parents)):
                    inj += p_weights[t] * parents[t].get_a()
                node.set_a(logistic(in_j))
    
    def back_propagate(self, output_lst):
        for j in len(self.out_nodes):
            # must actually calculate in_j
            in_j = 0
            
            curr_node = self.out_nodes[j]
            temp_value = logistic(in_j)/curr_node.get_a()
            curr_node.error = temp_value * (1 - temp_value) * (output_lst[j] - curr_node.get_a())
            
        # now do similar thing for mid-layer nodes
    



def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
