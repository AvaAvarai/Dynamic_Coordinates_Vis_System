import numpy as np
import MODEL


def compute_positions(data, df_name, section_array):
    # Use a copy of the dataframe to avoid changing the original data
    df_copy = df_name.copy()
    for index, is_inverted in enumerate(data.attribute_inversions):
        if is_inverted:
            df_copy.iloc[:, index] = 1 - df_copy.iloc[:, index]

    x_coord = np.tile(section_array, reps=len(df_copy.index))
    y_coord = df_copy.to_numpy().ravel()
    
    # Apply vertical shifts to the data points
    for i in range(len(data.axis_vertical_shifts)):
        shift_indices = range(i, len(y_coord), data.vertex_count)
        y_coord[shift_indices] += data.axis_vertical_shifts[i]
    
    pos_array = np.column_stack((x_coord, y_coord))
    return pos_array


def compute_axis_positions(data, section_array):
    axis_vertex_array = []
    for idx in range(data.vertex_count):
        # Add vertical shifts to the axis endpoints
        shift = data.axis_vertical_shifts[idx] if idx < len(data.axis_vertical_shifts) else 0
        axis_vertex_array.extend([
            [section_array[idx], 0 + shift],  # Bottom point
            [section_array[idx], 1 + shift]   # Top point
        ])
    return axis_vertex_array


class PC:
    def __init__(self, data: MODEL.Dataset):
        # Normalization using the MODEL class function
        data.dataframe = data.normalize_data(our_range=(0, 1))

        # Create section_array based on vertex_count
        section_array = np.linspace(start=0, stop=1, num=data.vertex_count)

        # Compute positions for each class and store in data.positions
        data.positions = []
        for class_name in data.class_names:
            df_name = data.dataframe[data.dataframe['class'] == class_name]
            df_name = df_name.drop(columns='class', axis=1)
            pos_array = compute_positions(data, df_name, section_array)
            data.positions.append(pos_array)

        # Compute axis positions
        data.axis_positions = compute_axis_positions(data, section_array)
        data.axis_count = data.vertex_count
