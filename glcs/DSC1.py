import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DSC1:
    def __init__(self, dataset):
        working_df = dataset.dataframe.copy()
        space = 1 / dataset.vertex_count
        scaler = MinMaxScaler((0, space))  # [0, 1 / vertex_count] scaling
        working_df[dataset.attribute_names] = scaler.fit_transform(working_df[dataset.attribute_names])

        angle_array = np.repeat(45, repeats=dataset.vertex_count)
        angle_array[0] = 80

        for name in dataset.class_names:
            df_name = working_df[working_df['class'] == name]
            df_name = df_name.drop(columns='class', axis=1)

            # positions
            values = df_name.to_numpy()
            values = values.ravel()

            j = 0
            scaffolds = np.asarray([[0, 0]])
            for i in range(len(df_name.index) * dataset.vertex_count):
                if i % dataset.vertex_count == 0:
                    j = 0
                    new_x = np.cos(np.deg2rad(angle_array[j])) * values[i]
                    new_y = np.sin(np.deg2rad(angle_array[j])) * values[i]
                    scaffolds = np.append(scaffolds, [[-1 + new_x, -1 + new_y]], 0)
                else:
                    j += 1
                    new_x = np.cos(np.deg2rad(angle_array[j])) * values[i]
                    new_y = np.sin(np.deg2rad(angle_array[j])) * values[i]
                    scaffolds = np.append(scaffolds, [
                        [scaffolds[i][0] + new_x, scaffolds[i][1] + new_y]], 0)

            scaffolds = np.delete(scaffolds, 0, 0)
            dataset.positions.append(scaffolds)

        dataset.axis_positions = [[-1, -1], [-1, 1], [-1, -1], [1, -1]]
        dataset.axis_count = 2
