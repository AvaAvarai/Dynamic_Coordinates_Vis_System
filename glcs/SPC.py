import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SPC:
    def __init__(self, dataset):
        working_df = dataset.dataframe.copy()

        section_array = np.linspace(start=-1, stop=1, num=dataset.vertex_count + 1)
        j = 0
        for i in range(dataset.attribute_count):
            if i % 2 != 0:
                scaler = MinMaxScaler((-1, 1))
            else:
                scaler = MinMaxScaler((section_array[j], section_array[j + 1]))
                j += 1

            # Apply the scaling
            working_df[working_df.columns[i]] = scaler.fit_transform(working_df[[working_df.columns[i]]])

        # Apply inversions to the scaled data if necessary
        for i in range(dataset.attribute_count):
            if dataset.attribute_inversions[i]:
                # Calculate the min and max from the data for each attribute
                attribute_min = working_df.iloc[:, i].min()
                attribute_max = working_df.iloc[:, i].max()

                # Calculate the midpoint
                midpoint = (attribute_max + attribute_min) / 2

                # Invert the attribute by reflecting it across the midpoint
                working_df.iloc[:, i] = (midpoint - (working_df.iloc[:, i] - midpoint))

        # After computing positions
        for name in dataset.class_names:
            df_name = working_df[working_df['class'] == name]
            df_name = df_name.drop(columns='class', axis=1)

            # Compute positions
            values = df_name.to_numpy()
            values = values.ravel()
            values = np.reshape(values, (-1, 2))

            dataset.positions.append(values)

        # Calculate axis positions for visualization
        axis_vertex_array = [[-1, -1], [1, -1]]
        for i in range(dataset.vertex_count):
            axis_vertex_array.append([section_array[i], -1])
            axis_vertex_array.append([section_array[i], 1])

        dataset.axis_positions = axis_vertex_array
        dataset.axis_count = dataset.vertex_count + 1
