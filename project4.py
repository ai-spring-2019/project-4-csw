"""
NAME:       Colleen Wahl
DATE:       May 13, 2019
PROJECT:    Project 4: Neural Network
HOW TO USE: On the command line, after the name of the data file, write the number
            of iterations you want the network to do to learn weights. After that,
            include the k value for k-fold cross validation (the # of subsets you
            want). You can type n in place of a number if you want to do "leave
            one out" validation.
            ex: python3 project4.py generated.csv 50 5
            (Note: for the 3-bit incrementer, set the k value to be 1. Also, if you
                   want to see any meaningful results for the incrementer, uncomment
                   the code so-noted in the accuracy function.)
            Any other modifications must be made in the main function, and
            there are specific directions as to how to modify those elements in
            comments in the main function.
"""


import csv, sys, random, math, copy


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

        # Comment the following code in to see results for the three-bit incrementer
        #outputs = nn.get_outputs()
        #print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)


class NetNode():
    """ A node in the neural network. """
    
    def __init__(self, a = 'None', error = 'None'):
        """ Initialize a node to store an activation value, an error value,
            a list of parent nodes, a list of children nodes, and the weights
            associated with all those connections. """
        
        self._a = a
        self.error = error
        self.classification = None
        self.parents = []
        self.p_weights = []
        self.children = []
        self.c_weights = []
    
    def set_a(self, a):
        """ Set the activation value. """
        
        self._a = a
        
    def get_a(self):
        """ Return the activation value. """
        
        return self._a
    

class NeuralNetwork():
    """ A neural network class. """
    
    def __init__(self, node_num_lst, decrease = False, alpha = 0.1):
        """ Initialize the neural network based on the given #s of nodes
            per layer. """
        
        self.alpha = alpha
        
        # determines whether alpha will get smaller as we go
        self.decrease = decrease

        # if we do have a decreasing alpha, set it initially to 1
        if decrease:
            self.alpha = 1

        # create list of input nodes
        self.in_nodes = []
        for _ in range(node_num_lst[0]):
            self.in_nodes.append(NetNode())
                         
        # create list to contain nodes in hidden layers and the output layer
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

        # store the output nodes in their own list, too, just to make things easier
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

    def get_outputs(self):
        """ Return the vector of activation values. """
        
        out_a_lst = []
        for node in self.out_nodes:
            out_a_lst.append(node.get_a())
        return out_a_lst

    def convert_to_vec(self, out_val):
        """ Return a vector version of the output data. """

        # if it is already a vector (ex: the bit incrementer), do nothing
        if len(out_val) > 1:
            return out_val

        out_num = len(self.out_nodes)
        vec = []
        for i in range(out_num):
            if i == out_val[0]:
                vec.append(1)
            else:
                vec.append(0)
                
        return vec
            
    def predict_class(self):
        """ Return which class this is part of. """
        
        output_ind = 0
        max_val = 0
        for i in range(len(self.out_nodes)):
            if self.out_nodes[i].get_a() > max_val:
                max_val = self.out_nodes[i].get_a()
                output_ind = i
        return output_ind
    
    def forward_propagate(self, input_lst):
        """ Given a list of inputs, determine all nodes' activations. """

        # set activation for input nodes to the corresponding input value
        for i in range(len(input_lst)):
            self.in_nodes[i].set_a(input_lst[i])

        # set activation values for non-input nodes
        for layer in self.other_nodes:
            for node in layer:
                in_j = 0
                for t in range(len(node.parents)):
                    in_j += node.p_weights[t] * node.parents[t].get_a()
                node.set_a(logistic(in_j))
    
    def back_propagate(self, output_item):
        """ Back propagate error values through the network. """

        output_lst = self.convert_to_vec(output_item)

        # determine error values for the output nodes
        for j in range(len(self.out_nodes)):
            curr_node = self.out_nodes[j]
            temp_value = curr_node.get_a()
            curr_node.error = temp_value * (1 - temp_value) * (output_lst[j] - curr_node.get_a())
            
        # now do similar thing for mid-layer nodes
        relevant_layers = self.other_nodes[:-1]
        relevant_layers = [self.in_nodes] + relevant_layers
        q = len(relevant_layers) - 1
        while q > -1:
            layer = relevant_layers[q]
            q-=1
            for i in range(len(layer)):
                curr_node = layer[i]
                g = curr_node.get_a()
                sum_thing = 0
                for t in range(len(curr_node.children)):
                    sum_thing += curr_node.c_weights[t] * curr_node.children[t].error
                curr_node.error = g * (1 - g) * sum_thing
                
    def update_weights(self):
        """ Update the weights based on the error values associated with each node. """
        
        # update parent weights
        for layer in self.other_nodes:
            for i in range(len(layer)):
                curr_node = layer[i]
                for t in range(len(curr_node.parents)):
                    curr_node.p_weights[t] = curr_node.p_weights[t] + self.alpha * curr_node.parents[t].get_a() * curr_node.error
                    
        # update children weights
        relevant_layers = self.other_nodes[:-1]
        relevant_layers.append(self.in_nodes)
        for layer in relevant_layers:
            for i in range(len(layer)):
                curr_node = layer[i]
                for t in range(len(curr_node.children)):
                    curr_node.c_weights[t] = curr_node.c_weights[t] + self.alpha * curr_node.get_a() * curr_node.children[t].error
                    
    def run_network(self, test_data, iters):
        """ Run the network for the given number of iterations. """
        
        for i in range(iters):
            for (x, y) in test_data:
                self.forward_propagate(x)
                self.back_propagate(y)
                self.update_weights()

            # if we are using a decreasing alpha, decrease it
            if self.decrease:
                self.alpha = 1000/(1000 + i)
                              
    
def main():
    """ Create, train, and test a neural network. """
    
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Figure out how many outputs we have.
    distinct_outputs = set()
    for example in training:
        if example[1][0] not in distinct_outputs:
            distinct_outputs.add(example[1][0])

    out_num = len(distinct_outputs)

    # but if we are not doing the three-bit incrementer,
    # override that with the size of the vector.
    if len(training[0][1]) > 1:
        out_num = len(training[0][1])

    # get the number of nodes we should have in the input layer
    in_num = len(training[0][0])

    #print(in_num, out_num)

    # determine what the k value is for k-fold cross-validation
    if sys.argv[3] == 'n':
        k_val = len(training)
    else:
        k_val = int(sys.argv[3])

    # use the k value to determine the size of subsets
    set_size = len(training)//k_val

    # create a list of subsets
    pairs_copy = copy.deepcopy(training)
    random.shuffle(pairs_copy)
    subsets = []
    for _ in range(k_val - 1):
        subsets.append(pairs_copy[:set_size])
        pairs_copy = pairs_copy[set_size:]
    subsets.append(pairs_copy)

    accuracy_sum = 0

    # do k training/test runs using a different subset
    # for testing each time
    for i in range(k_val):
        print(i)
        # choose the subset for testing
        chosen_one = subsets[i]

        # create a list containing all the other test cases
        train_sets = []
        for k in range(k_val):
            if k != i:
                train_sets = train_sets + subsets[k]
                
        # in the special case where k = 1, set the training set equal to the testing set.
        if k_val == 1:
            train_sets = chosen_one

        # set up the neural network.
        # DESCRIPTION OF PARAMETERS:
        #   1. the first parameter is a list that determines how
        #      many nodes go in each layer. The # of nodes in the input
        #      and output layer are pre-calculated, so it is just the
        #      hidden layers that you can define. Ex: if you want a network
        #      with 2 hidden layers where the first has 3 nodes and the second
        #      has 6, it would look like [in_num, 3, 6, out_num]
        #   2. The second parameter is optional (default False). This determines
        #      if you want to use a decreasing alpha value. (False = no decrease).
        #   3. The third parameter is also optional (default = 0.1). This sets the
        #      initial alpha value (although it will only have an effect if you
        #      are using a non-decreasing alpha).
        nn = NeuralNetwork([in_num, 6, out_num], False)

        # train the nerual network on the selected examples
        nn.run_network(train_sets, int(sys.argv[2]))

        # test the neural network for accuracy on the selected examples
        accuracy_sum += accuracy(nn, chosen_one)

    # output the averaged accuracy
    print("Accuracy:", accuracy_sum/k_val)

    
if __name__ == "__main__":
    main()
