from typing import List, Optional, Tuple
import WARNINGS
import pandas as pd
import numpy as np
import os
import random
from ctgan import CTGAN
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import COLORS


class Dataset:
    def __init__(self):
        # dataset info
        self.name: str = ''
        self.filepath: str = ''
        
        self.dataframe: Optional[pd.DataFrame] = None

        self.not_normalized_frame: Optional[pd.DataFrame] = None

        # class information
        self.class_count: int = 0
        self.count_per_class: List[int] = []
        self.class_names: List[str] = []
        self.class_colors: List[Tuple[int, int, int, int]] = []

        self.rule_regions = {}

        # attribute information
        self.attribute_count: int = 0
        self.attribute_names: List[str] = []
        self.attribute_alpha: int = 255  # for attribute slider

        self.attribute_inversions: List[bool] = []  # for attribute inversion option
        self.overlap_points = {}

        # sample information
        self.sample_count: int = 0
        self.clipped_samples: np.ndarray = np.array([], dtype=float)  # for line clip option
        self.clear_samples: np.ndarray = np.array([], dtype=float)
        self.vertex_in: np.ndarray = np.array([], dtype=float)  # for vertex clip option
        self.last_vertex_in: np.ndarray = np.array([], dtype=float)  # for last vertex clip option

        # plot information
        self.plot_type: str = ''
        self.positions: List[float] = []
        
        self.overlap_indices = []
        self.radial_bounds = {}
        
        self.axis_positions: List[float] = []
        self.axis_on: bool = True
        self.axis_count: int = 0
        self.vertex_count: int = 0

        self.trace_mode: bool = False

        self.coefs = []
        self.fitted = False

        self.active_attributes: np.ndarray = np.array([], dtype=bool)
        self.active_classes: List[bool] = []
        self.active_markers: List[bool] = []
        self.active_sectors: List[bool] = []
        
        self.class_order: List[int] = []
        self.attribute_order: List[int] = []
        self.all_arc_lengths: List[int] = []

    def duplicate_last_attribute(self):
        if self.dataframe is None or self.dataframe.empty:
            print("DataFrame is not loaded or is empty.")
            return

        last_attribute = self.dataframe.columns[-2]
        new_attribute = f'{last_attribute}_copy'
        self.dataframe[new_attribute] = self.dataframe[last_attribute]
        self.attribute_names.append(new_attribute)
        self.attribute_count += 1
        self.vertex_count += 1
        self.active_attributes = np.append(self.active_attributes, True)
        self.attribute_inversions = np.append(self.attribute_inversions, False)
        
        # reorder DataFrame columns to ensure 'class' is the last column
        cols = self.dataframe.columns.tolist()
        cols.append(cols.pop(cols.index('class')))  # Move 'class' to the end
        self.dataframe = self.dataframe[cols]

    def reload(self):
        if self.filepath:
            try:
                # store inversions
                inversions = self.attribute_inversions
                df = pd.read_csv(self.filepath)
                self.load_frame(df)
                # restore inversions
                self.attribute_inversions = inversions
            except Exception as e:
                print(f"Error reloading data: {e}")
        else:
            print("No filepath set for reloading.")

    def inject_datapoint(self, data_point: List[float], class_name: str):
        self.dataframe = self.dataframe._append(pd.Series(data_point + [class_name], index=self.dataframe.columns), ignore_index=True)
        self.not_normalized_frame = self.not_normalized_frame._append(pd.Series(data_point + [class_name], index=self.not_normalized_frame.columns), ignore_index=True)
        self.sample_count += 1
        self.count_per_class[self.class_names.index(class_name)] += 1
        # update clipped_samples array with new sample
        self.clipped_samples = np.append(self.clipped_samples, False)
        self.clear_samples = np.append(self.clear_samples, False)

    def update_coef(self, attribute_index, new_coef_value):
        if 0 <= attribute_index < len(self.coefs):
            self.coefs[attribute_index] = new_coef_value

    def load_frame(self, df: pd.DataFrame, not_normal=None):
        # put class column to end of dataframe
        df.insert(len(df.columns) - 1, 'class', df.pop('class'))

        # get class information
        self.class_count = len(df['class'].unique())
        self.class_names = df['class'].unique().tolist()  # Keep unique class names in their original order
        self.count_per_class = [df['class'].tolist().count(name) for name in self.class_names]
        self.class_order = np.arange(0, self.class_count)

        # get class colors and lower case
        names = [str(name).lower() for name in self.class_names]
        gen_green_red = False
        for name in names:
            if name in ['benign', 'malignant', 'positive', 'negative']:
                gen_green_red = True
        if gen_green_red:
            self.class_colors = COLORS.getColors(self.class_count, [0, 0, 0], [1, 1, 1], names, benign_malignant=True).colors_array
        else:
            self.class_colors = COLORS.getColors(self.class_count, [0, 0, 0], [1, 1, 1], names).colors_array

        # initialize arrays for class options
        self.active_markers = np.repeat(True, self.class_count)
        self.active_classes = np.repeat(True, self.class_count)
        self.active_sectors = np.repeat(True, self.class_count)

        # get attribute information
        self.attribute_names = df.columns.tolist()[:-1]
        self.attribute_count = len(df.columns) - 1
        self.attribute_order = np.arange(0, self.attribute_count)
        self.max_radial_distances = [0] * self.attribute_count
        self.coefs = np.ones(self.attribute_count) * 100

        self.active_attributes = np.repeat(True, self.attribute_count)
        self.attribute_inversions = np.repeat(False, self.attribute_count)

        # get sample information
        self.sample_count = len(df.index)
        # initialize arrays for clipping options
        self.clipped_samples = np.repeat(False, self.sample_count)
        self.clear_samples = np.repeat(False, self.sample_count)
        self.vertex_in = np.repeat(False, self.sample_count)
        self.last_vertex_in = np.repeat(False, self.sample_count)

        # general dataframe
        self.dataframe = df

        if not_normal is not None:
            self.not_normalized_frame = not_normal
        else:
            self.not_normalized_frame = df.copy()

    def delete_clip(self):
        """Delete the selected samples from the dataframe."""
        if self.dataframe is None or self.dataframe.empty:
            print("DataFrame is not loaded or is empty.")
            return

        if not any(self.clipped_samples):
            print("No samples selected for deletion.")
            return

        # Create a boolean mask for rows to be deleted
        bool_clipped = np.array(self.clipped_samples, dtype=bool)

        # Drop the rows from both dataframes
        self.dataframe = self.dataframe.loc[~bool_clipped].reset_index(drop=True)
        self.not_normalized_frame = self.not_normalized_frame.loc[~bool_clipped].reset_index(drop=True)

        # Update class information
        self.sample_count = len(self.dataframe.index)
        self.count_per_class = [self.dataframe['class'].tolist().count(name) for name in self.class_names]

        # Initialize arrays for clipping options
        self.clipped_samples = np.repeat(False, self.sample_count)
        self.clear_samples = np.repeat(False, self.sample_count)
        self.vertex_in = np.repeat(False, self.sample_count)
        self.last_vertex_in = np.repeat(False, self.sample_count)

        # Preserve the current class colors mapping
        class_color_mapping = dict(zip(self.class_names, self.class_colors))

        # Reload the frame to ensure consistency
        self.load_frame(self.dataframe, self.not_normalized_frame)

        # Restore the preserved class colors mapping
        self.class_colors = [class_color_mapping[class_name] for class_name in self.class_names]

    def copy_clip(self):
        """
        Clone cases with the same values and class colors as the selected (clipped) cases.
        """
        if self.dataframe is None or self.dataframe.empty:
            print("DataFrame is not loaded or is empty.")
            return
        if not any(self.clipped_samples):
            print("No samples selected for cloning.")
            return

        # Identify clipped samples
        clipped_indices = np.where(self.clipped_samples)[0]
        if len(clipped_indices) == 0:
            print("No clipped indices found.")
            return

        # Filter rows to clone
        cloned_rows = self.dataframe.iloc[clipped_indices].copy()

        # Duplicate the selected rows
        for i in range(len(cloned_rows)):
            new_row = cloned_rows.iloc[i]
            self.dataframe = self.dataframe._append(new_row, ignore_index=True)
            self.not_normalized_frame = self.not_normalized_frame._append(new_row, ignore_index=True)

        # Update sample count and clipping arrays
        self.sample_count = len(self.dataframe)
        # update the class counts
        self.count_per_class = [self.dataframe['class'].tolist().count(name) for name in self.class_names]
        self.clipped_samples = np.append(self.clipped_samples, [False] * len(cloned_rows))
        self.clear_samples = np.append(self.clear_samples, [False] * len(cloned_rows))

    def generate_data(self, num_samples: int, epochs: int):
        """Generate a specified number of samples using CTGAN."""
        # Initialize CTGAN
        ctgan = CTGAN(epochs=epochs, verbose=True)

        # Separate features and labels
        features = self.dataframe[self.attribute_names]
        labels = self.dataframe['class']

        # Encode categorical labels as integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Concatenate features and encoded labels
        features_with_labels = features.copy()
        features_with_labels['class_encoded'] = encoded_labels

        # Train CTGAN
        discrete_columns = ['class_encoded']  # Specify discrete columns
        ctgan.fit(features_with_labels, discrete_columns)

        # Generate synthetic samples
        synthetic_data = ctgan.sample(num_samples)

        # Decode labels back to original class names
        synthetic_data['class_encoded'] = label_encoder.inverse_transform(synthetic_data['class_encoded'])
        synthetic_features = synthetic_data.drop(columns=['class_encoded'])
        synthetic_labels = synthetic_data['class_encoded']

        # Inject synthetic samples into the dataframe
        for i in range(len(synthetic_features)):
            self.inject_datapoint(synthetic_features.iloc[i].tolist(), synthetic_labels.iloc[i])

    def move_samples(self, move_delta: int):
        """Move the selected samples up or down in the dataframe."""
        if self.dataframe is None or self.dataframe.empty:
            print("DataFrame is not loaded or is empty.")
            return

        if not any(self.clipped_samples):
            print("No samples selected.")
            return

        if move_delta == 0:
            return

        bool_clipped = np.array(self.clipped_samples, dtype=bool)
        
        for attribute in self.attribute_names:
            normalized_range = self.dataframe[attribute].max() - self.dataframe[attribute].min()
            if normalized_range == 0:
                continue  # Skip attributes with no variation

            proportional_delta = move_delta / normalized_range
            
            # Calculate the proportional move for the normalized and not_normalized frames
            if self.dataframe.loc[bool_clipped, attribute].min() + proportional_delta > 0 and self.dataframe.loc[bool_clipped, attribute].max() + proportional_delta < 1:
                not_normalized_range = self.not_normalized_frame[attribute].max() - self.not_normalized_frame[attribute].min()
                if not_normalized_range == 0:
                    continue  # Skip attributes with no variation

                not_normalized_proportional_delta = proportional_delta * not_normalized_range
                self.not_normalized_frame[attribute] = self.not_normalized_frame[attribute].astype(float)
                self.dataframe.loc[bool_clipped, attribute] += proportional_delta
                self.not_normalized_frame.loc[bool_clipped, attribute] += not_normalized_proportional_delta

    def load_from_csv(self, filename: str):
        """Load the dataset from a CSV file."""
        try:
            df = pd.read_csv(filename)
            self.name = os.path.basename(filename)
            self.filepath = filename
            self.load_frame(df)

        except Exception as e:
            print(f"An error occurred: {e}")
            
    def normalize_data(self, our_range: Tuple[float, float]):
        """Normalize the data in the dataframe to the specified range."""
        if self.dataframe is None or self.dataframe.empty:
            print("DataFrame is not loaded or is empty.")
            return self.dataframe

        scaler = MinMaxScaler(our_range)
        # Only normalize self.dataframe
        self.dataframe[self.attribute_names] = scaler.fit_transform(self.dataframe[self.attribute_names])
        return self.dataframe

    def normalize_col(self, col: int, our_range: Tuple[float, float]):
        """Normalize a specific column in the dataframe to the specified range."""
        scaler = MinMaxScaler(our_range)
        self.dataframe[self.attribute_names[col]] = scaler.fit_transform(self.dataframe[[self.attribute_names[col]]])
        return self.dataframe

    def roll_clips(self, roll_dir: int):
        """Select the next sample(s) to clip"""
        self.clipped_samples = list(np.roll(self.clipped_samples, roll_dir))

    def roll_vertex_in(self, roll_dir: int):
        """Select the previous sample(s) to clip"""
        self.vertex_in = list(np.roll(self.vertex_in, roll_dir))
