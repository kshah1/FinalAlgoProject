import math
import sys

def EntropyAndError(inputs, outputs):
    with open(outputs, "w") as o:
        with open(inputs, "r") as i:
            labels = {}
            for idx, line in enumerate(i.readlines()):
                if idx == 0:
                    pass
                else:
                    label = line.strip().split(',')[-1]
    
                    if label not in labels:
                        labels[label] = 1
                    else:
                        labels[label] += 1
                        
                
        key_one, key_two = labels.keys()
        total = labels[key_one] + labels[key_two]
    
        #error rate
        if labels[key_one] > labels[key_two]:
            error = labels[key_two]/total
        elif labels[key_one] < labels[key_two]:
            error = labels[key_one]/total
        else:
            error = 0
        
        #entropy
        entropy = -1 * ((labels[key_one] / total) * (math.log2(labels[key_one] / total)) + (labels[key_two]/total) * (math.log2(labels[key_two] / total)))
            
        o.write("entropy: {}\n".format(entropy))
        o.write("error: {}".format(error))

if __name__ == "__main__":
    inputs = sys.argv[1]
    outputs = sys.argv[2]
    EntropyAndError(inputs, outputs)