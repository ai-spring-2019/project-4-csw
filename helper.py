# to help normalize data and create csv files in the right format

import math, csv

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

header, data = read_data("poker1.csv", "\t")
pairs = convert_data_to_pairs(data, header)

for i in range(len(pairs)):
    new_y = 0
    for j in range(len(pairs[i][1])):
        if pairs[i][1][j] != 0:
            new_y = j
    pairs[i] = (pairs[i][0], [new_y])

x_avgs = []
x_std_devs = []
for i in range(len(pairs[0][0])):
    x_avgs.append(0)
    for (x, y) in pairs:
        x_avgs[i] += x[i]
    x_avgs[i] = x_avgs[i]/len(pairs)
        
    
for i in range(len(pairs[0][0])):
    temp_sum = 0
    for (x, y) in pairs:
        val = x[i] - x_avgs[i]
        val = val * val
        temp_sum += val
    x_std_devs.append(math.sqrt(temp_sum/len(pairs)))

new_x_vals = []
for _ in range(len(pairs)):
    new_x_vals.append([])
    
for i in range(len(pairs[0][0])):
    for j in range(len(new_x_vals)):
        new_x_vals[j].append((pairs[j][0][i] - x_avgs[i])/x_std_devs[i])

f = open("poker2.csv", "w")
f.write(header[0])
header=header[1:]
for item in header:
    f.write("," + item)
f.write("\n")
for i in range(len(pairs)):
    for item in new_x_vals[i]:
        f.write(str(item) + ",")
    f.write(str(pairs[i][1][0]) + "\n")

f.close()
        
