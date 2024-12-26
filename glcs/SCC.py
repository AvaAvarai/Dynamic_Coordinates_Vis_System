import numpy as np
import DATASET


def compute_coordinates(data, df, class_index):
    base_radius = (data.attribute_count / (2 * np.pi))

    # Adjust the radius based on class index
    if class_index < 2:
        # First two classes share the first axis
        radius_factor = 1
    else:
        # Subsequent classes each get their own axis, scaling geometrically
        scale_factor = 2.1  # Adjust this factor to control the rate of radius increase
        radius_factor = scale_factor * (class_index - 1)

    radius = base_radius * radius_factor
    section_array = np.linspace(0, 1, data.attribute_count)
    x_coord = np.tile(section_array, reps=len(df.index))
    y_coord = df.to_numpy().ravel()

    attribute_length = data.attribute_count * len(df.index)
    arc_length = 0
    arc_rule = 0

    for i in range(attribute_length):
        if arc_rule >= data.attribute_count:
            arc_length = 0
            arc_rule = 0

        if data.attribute_inversions[arc_rule]:
            arc_length += (1 - y_coord[i])
        else:
            arc_length += y_coord[i]

        # Apply the radius factor to the angle calculation
        center_angle = arc_length * 360 / (2 * np.pi * radius) * radius_factor
        center_angle = np.pi * center_angle / 180

        x_coord[i] = radius * np.sin(center_angle)
        y_coord[i] = radius * np.cos(center_angle)

        arc_rule += 1
        arc_length = arc_rule

    return np.column_stack((x_coord, y_coord))


class SCC:
    def __init__(self, data: DATASET.Dataset):
        data.vertex_count = data.attribute_count
        data.dataframe = data.normalize_data(our_range=(0, 1))

        # Compute coordinates for each class with adjusted radius
        data.positions = [
            compute_coordinates(data, data.dataframe[data.dataframe['class'] == class_name].drop('class', axis=1),
                                class_index)
            for class_index, class_name in enumerate(data.class_names)]

        data.axis_count = data.attribute_count
