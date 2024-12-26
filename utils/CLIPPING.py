import numpy as np
import pandas as pd
from csv import writer
import os
import DATASET

# exit codes for cohen-sutherland
INSIDE = 0  # 0000b
LEFT   = 1  # 0001b
RIGHT  = 2  # 0010b
BOTTOM = 4  # 0100b
TOP    = 8  # 1000b

# Vertex clip: check if the vertex is inside the rectangle
# Line clip: check if the line is inside the rectangle
# End clip: check if the last vertex of the line is inside the rectangle
def clip_display(textbox, dataset):
    total_sample = dataset.sample_count
    filenames = ['test_line.csv', 'test_vertex.csv', 'test_end.csv']
    clip_types = ['Line Clip', 'Vertex Clip', 'End Clip']
    info_string = ''

    for cnt, file in enumerate(filenames):
        df = pd.read_csv(file)
        # get dataset name
        name = os.path.basename(file)

        # get class information
        class_names = df['class'].value_counts().index.tolist()
        count_per_class = df['class'].value_counts().tolist()
        sample_count = len(df.index)

        # display class data
        info_string += ('Clip Type: ' + clip_types[cnt] + '  (Output File: ' + name + ')\nTotal Case Count: ' + str(sample_count) + '/' + str(total_sample) + ' ({:.2f}'.format(sample_count / total_sample * 100)) + '%)\n'

        # loop through class names
        for counter, ele in enumerate(class_names):
            if counter == 0:
                info_string += '\n'
            info_string += ('Class ' + str(counter+1) + ': ' + str(ele) + '\n' + 'Class Case Count: ' + str(count_per_class[counter]) + '/' + str(dataset.count_per_class[counter]) + ' ({:.2f}'.format(count_per_class[counter] / dataset.count_per_class[counter] * 100) + '%)\n')
        info_string += '\n'

    textbox.setText(info_string)


def clip_files(dataset, textbox):
    # for option 1 (lines clipped)
    output_train1 = open('train_line.csv', 'w', newline='')
    output_test1 = open('test_line.csv', 'w', newline='')
    csv_writer_train1 = writer(output_train1)
    csv_writer_test1 = writer(output_test1)

    # for option 2 (vertex bounded)
    output_train2 = open('train_vertex.csv', 'w', newline='')
    output_test2 = open('test_vertex.csv', 'w', newline='')
    csv_writer_train2 = writer(output_train2)
    csv_writer_test2 = writer(output_test2)

    # for option 3 (last vertex (line end) bounded)
    output_train3 = open('train_end.csv', 'w', newline='')
    output_test3 = open('test_end.csv', 'w', newline='')
    csv_writer_train3 = writer(output_train3)
    csv_writer_test3 = writer(output_test3)

    csv_writer_train1.writerow(np.append(np.array(dataset.attribute_names), 'class'))
    csv_writer_test1.writerow(np.append(np.array(dataset.attribute_names), 'class'))

    csv_writer_train2.writerow(np.append(np.array(dataset.attribute_names), 'class'))
    csv_writer_test2.writerow(np.append(np.array(dataset.attribute_names), 'class'))

    csv_writer_train3.writerow(np.append(np.array(dataset.attribute_names), 'class'))
    csv_writer_test3.writerow(np.append(np.array(dataset.attribute_names), 'class'))

    # build files
    for i in range(dataset.sample_count):
        if dataset.clipped_samples[i]:
            csv_writer_test1.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))
        else:
            csv_writer_train1.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))

        if dataset.vertex_in[i]:
            csv_writer_test2.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))
        else:
            csv_writer_train2.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))

        if dataset.last_vertex_in[i]:
            csv_writer_test3.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))
        else:
            csv_writer_train3.writerow(np.array(dataset.dataframe.iloc[[i]].squeeze()))

    output_train1.close()
    output_test1.close()

    output_train2.close()
    output_test2.close()

    output_train3.close()
    output_test3.close()

    # build text box
    clip_display(textbox, dataset)


class MinAndMax(object):
    x_min = None
    x_max = None
    y_min = None
    y_max = None


# Function to compute region code for a point(x, y)
def compute_code(x, y, min_max):
    code = INSIDE
    if x < min_max.x_min:  # to the left of rectangle
        code |= LEFT
    elif x > min_max.x_max:  # to the right of rectangle
        code |= RIGHT

    if y < min_max.y_min:  # below the rectangle
        code |= BOTTOM
    elif y > min_max.y_max:  # above the rectangle
        code |= TOP

    return code


def cohen_sutherland_clip(x1, y1, x2, y2, min_max):
    # Compute region codes for P1, P2
    code1 = compute_code(x1, y1, min_max)
    code2 = compute_code(x2, y2, min_max)
    accept = False

    while True:
        # If both endpoints lie within rectangle
        if code1 == 0 and code2 == 0:
            accept = True
            break

        # If both endpoints are outside rectangle
        elif (code1 & code2) != 0:
            break

        # Some segment lies within the rectangle
        else:
            x = 1.0
            y = 1.0
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            if code_out & TOP:
                x = x1 + (x2 - x1) * (min_max.y_max - y1) / (y2 - y1)
                y = min_max.y_max

            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * (min_max.y_min - y1) / (y2 - y1)
                y = min_max.y_min

            elif code_out & RIGHT:
                y = y1 + (y2 - y1) * (min_max.x_max - x1) / (x2 - x1)
                x = min_max.x_max

            elif code_out & LEFT:
                y = y1 + (y2 - y1) * (min_max.x_min - x1) / (x2 - x1)
                x = min_max.x_min

            if code_out == code1:
                x1 = x
                y1 = y
                code1 = compute_code(x1, y1, min_max)
            else:
                x2 = x
                y2 = y
                code2 = compute_code(x2, y2, min_max)

    if accept:
        return True
    else:
        return False

def count_clipped_samples(dataset):
    clipped_count = 0
    
    for i in range(dataset.sample_count):
        if dataset.clipped_samples[i]:
            clipped_count += 1
    
    return clipped_count

def count_clipped_classes(dataset, skips=[]):
    clipped_classes = set()
    
    for i in range(dataset.sample_count):
        if dataset.clipped_samples[i] and not i in skips:
            clipped_classes.add(dataset.dataframe['class'][i])
    
    return clipped_classes

def primary_clipped_class(dataset):
    clipped_classes = count_clipped_classes(dataset)
    
    if len(clipped_classes) == 0:
        return None
    
    primary_class = None
    primary_class_count = 0
    
    for c in clipped_classes:
        class_count = len(dataset.dataframe[dataset.dataframe['class'] == c])
        
        if class_count > primary_class_count:
            primary_class = c
            primary_class_count = class_count
    
    return primary_class

def is_pure_class(dataset):
    clipped_classes = count_clipped_classes(dataset)
    
    return len(clipped_classes) == 1

def vertex_check(vertex_x, vertex_y, min_max):
    if (min_max.x_min <= vertex_x <= min_max.x_max) and (min_max.y_min <= vertex_y <= min_max.y_max):
        return True
    else:
        return False

def clip_samples(positions, rect, dataset):
    min_max = MinAndMax()
    min_max.x_min = min(rect[0], rect[2])
    min_max.x_max = max(rect[0], rect[2])
    min_max.y_min = min(rect[1], rect[3])
    min_max.y_max = max(rect[1], rect[3])

    cnt = 0
    class_num = 1
    # check each class
    for data_class in positions:
        # check each sample in the class
        for sample in data_class:
            # check each polyline in the line
            for i in range(dataset.vertex_count - 1):
                # Ensure the next point in the polyline exists before attempting to clip
                if 2 * i + 3 < len(sample):
                    is_clipped = cohen_sutherland_clip(sample[2 * i], sample[2 * i + 1], sample[2 * i + 2], sample[2 * i + 3], min_max)
                    if is_clipped:
                        dataset.clipped_samples[cnt] = True
                # Regardless of clipping, check if the current vertex is inside
                if 2 * i + 1 < len(sample):
                    is_inside = vertex_check(sample[2 * i], sample[2 * i + 1], min_max)
                    if is_inside and cnt < len(dataset.vertex_in):
                        dataset.vertex_in[cnt] = True
                # Special case for the last vertex
                if 2 * i + 3 < len(sample):
                    if i == dataset.vertex_count - 2:
                        last_is_inside = vertex_check(sample[2 * i + 2], sample[2 * i + 3], min_max)
                        if last_is_inside:
                            dataset.vertex_in[cnt] = True
                            dataset.last_vertex_in[cnt] = True
            cnt += 1
        class_num += 1


class Clipping:
    def __init__(self, rect, dataset):
        positions = []
        for i in range(dataset.class_count):
            positions.append(np.array(dataset.positions[i]))
            num_columns = dataset.vertex_count * 2
            num_rows = positions[i].size // num_columns
            positions[i] = np.reshape(positions[i], (num_rows, num_columns))


        clip_samples(positions, rect, dataset)

