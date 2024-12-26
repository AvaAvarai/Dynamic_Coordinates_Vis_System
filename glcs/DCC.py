import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class DCC:
    def __init__(self, dataset):
        dataset.minmax_arc_lengths = []
        
        working_df = dataset.dataframe.copy()
        scaler = MinMaxScaler((0, 1))
        working_df[dataset.attribute_names] = scaler.fit_transform(working_df[dataset.attribute_names])

        dataset.normalized_dataframe = working_df

        working_frame = working_df.copy()

        # strip last column
        working_frame = working_frame.iloc[:, 0:-1]
        
        scaler = MinMaxScaler((0, 1))
        attributes_scaled = scaler.fit_transform(working_df[dataset.attribute_names])

        # Prepare the data for LDA
        X = attributes_scaled
        y = working_df['class'].values
        
        # Fit LDA model first plot of DCC only per data loaded
        if not dataset.fitted: 
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            lda_coefs = np.abs(lda.coef_).mean(axis=0)
            dataset.fitted = True
            dataset.coefs = lda_coefs
            # sort the attributes by the coefficients
            sorted_indices = np.argsort(lda_coefs)
            dataset.attribute_names = list(np.array(dataset.attribute_names)[sorted_indices])
            dataset.attribute_order = sorted_indices
            dataset.coefs = dataset.coefs[sorted_indices]
        
        coefArr = dataset.coefs / 100
        
        for index, col in enumerate(working_frame.columns):
            columnCoef = coefArr[index]
            working_df[col] = columnCoef * working_df[col]

        section_array = np.linspace(start=0, stop=1, num=dataset.vertex_count)

        for class_index, class_name in enumerate(dataset.class_names):
            base_radius = (dataset.attribute_count / (2 * np.pi))

            # Adjust the radius based on class index
            if class_index < 2:
                # First two classes share the first axis
                radius_factor = 1
            else:
                # Subsequent classes each get their own axis, scaling geometrically
                scale_factor = 2.1  # Adjust this factor to control the rate of radius increase
                radius_factor = scale_factor * (class_index - 1)

            radius = base_radius * radius_factor

            df_name = working_df[working_df['class'] == class_name]
            df_name = df_name.drop(columns='class', axis=1)

            x_coord = np.tile(section_array, reps=len(df_name.index))
            y_coord = df_name.to_numpy().ravel()

            MinE = dataset.attribute_count
            MaxE = 0

            arc_length = 0
            arc_rule = 0

            attribute_length = dataset.attribute_count * dataset.count_per_class[class_index]
            for i in range(attribute_length):
                if arc_rule >= dataset.attribute_count:
                    arc_length = 0
                    arc_rule = 0

                arc_length += y_coord[i]

                center_angle = arc_length * 360 / (2 * np.pi * radius) * radius_factor
                center_angle = np.pi * center_angle / 180

                x_coord[i] = radius * np.sin(center_angle)
                y_coord[i] = radius * np.cos(center_angle)

                arc_rule += 1

            dataset.minmax_arc_lengths.append(MinE)
            dataset.minmax_arc_lengths.append(MaxE)

            pos_array = np.column_stack((x_coord, y_coord))
            dataset.positions.append(pos_array)

        dataset.axis_count = dataset.attribute_count
