import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DSC2:
    def __init__(self, dataset):
        working_df = dataset.dataframe.copy()
        scaler = MinMaxScaler((0, 1)) # [0, 1] scaling
        for i in range(dataset.attribute_count):
            working_df[working_df.columns[i]] = scaler.fit_transform(working_df[[working_df.columns[i]]])

        space_array = np.repeat(0.05, repeats=dataset.attribute_count)
        space_array[0] = 1
        space_array[1] = 1

        for i in range(dataset.attribute_count):
            scaler = MinMaxScaler((0, space_array[i]))
            working_df[working_df.columns[i]] = scaler.fit_transform(working_df[[working_df.columns[i]]])

        angle_array = np.repeat(0, dataset.vertex_count)

        for name in dataset.class_names:
            df_name = working_df[working_df['class'] == name]
            df_name = df_name.drop(columns='class', axis=1)

            # positions
            values = df_name.to_numpy()
            values = values.ravel()
            values = np.reshape(values, (-1, 2))
            scaffolds = np.asarray([[0, 0]])

            j = 0
            for i in range(len(df_name.index) * dataset.vertex_count):
                if i % dataset.vertex_count == 0:
                    j = 0
                    angle = np.deg2rad(angle_array[j]) # angle in radians
                    new_x = np.cos(angle) * values[i][0] - np.sin(angle) * values[i][1]
                    new_y = np.sin(angle) * values[i][0] + np.cos(angle) * values[i][1]
                    scaffolds = np.append(scaffolds, [[-1 + new_x, -1 + new_y]], 0)
                    j += 1
                else:
                    angle = np.deg2rad(angle_array[j]) # angle in radians
                    new_x = np.cos(angle) * values[i][0] - np.sin(angle) * values[i][1]
                    new_y = np.sin(angle) * values[i][0] + np.cos(angle) * values[i][1]
                    j += 1
                    scaffolds = np.append(scaffolds, [[scaffolds[i][0] + new_x, scaffolds[i][1] + new_y]], 0)

            scaffolds = np.delete(scaffolds, 0, 0)
            dataset.positions.append(scaffolds)

        dataset.axis_positions = [[-1, -1], [-1, 1], [-1, -1], [1, -1]]
        dataset.axis_count = 2
