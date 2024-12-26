# DCVis: Dynamic Coordinate Visualization System

DCVis allows users to interactively build visual ML models for classifying new data. It offers tools to explore multidimensional data using General Line Coordinate visualizations and includes features for interactive classification and synthetic data generation to enhance supervised learning models. Synthetic samples can be tested against standard KNN, SVM, and Naive Bayes classifiers.

Experiments with DCVis have demonstrated insufficiencies in synthetic data generation and automated data labeling methods such as SMOTE, which are critical for the development of AI/ML models due to the need for large datasets for training. We have developed a new method for synthetic data generation and automated data labeling which is described in [2] using the formerly developed visualization methods for model development from [3]. Next, we are developing new methods for defining the class boundaries in multidimensional data using interactive visual ML [1].

GAN used for naive synthetic data generation is not developed by us, available at [https://github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN).

## Demo Video

Quick 3 minute video [here](https://youtu.be/a0F5xT2oiZ8).

## Requirements

- Python 3.x: Recommended version 3.6 or higher (primarily developed on Python 3.10.6).
- Dependencies: Install libraries from requirements.txt by running `pip install -r requirements.txt` in the terminal.

## Setup

To set up the project:

1. **Download the project and open it in a terminal:**
   - Use git to download and open in an IDE with an integrated terminal.
2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   ```

   (The virtual environment will be named `venv`)
3. **Activate the virtual environment:**
   (Refer to the table below for your OS and shell)
4. **Install the required libraries:**

   ```sh
   python -m pip install -r requirements.txt
   ```

5. **Start the application:**

   ```sh
   python DCVIS_MAIN.py
   ```

| Platform   | Shell     | Command to activate virtual environment |
|------------|-----------|-----------------------------------------|
| POSIX      | bash/zsh  | `$ source <venv>/bin/activate`          |
|            | fish      | `$ source <venv>/bin/activate.fish`     |
|            | csh/tcsh  | `$ source <venv>/bin/activate.csh`      |
| PowerShell |           | `$ <venv>/bin/Activate.ps1`             |
| Windows    | cmd.exe   | `C:\> <venv>\Scripts\activate.bat`      |
|            | PowerShell| `PS C:\> <venv>\Scripts\Activate.ps1`   |

Source: [Python Virtual Environments Documentation](https://docs.python.org/3/library/venv.html)

## Running

Execute the `DCVIS_MAIN.py` script to launch the DCVis application:

```sh
python DCVIS_MAIN.py
```

## Software Features

DCVis offers tools for visualizing and analyzing multidimensional numerical data:

## Visualization Capabilities

- **2D Visualizations**: Use OpenGL to render multidimensional data from .csv or .txt files.
- **Color Palette Generation**: Generates default palette colors with most possibly distinct colors.  
('Benign' and 'Positive' class names get green by default and 'Malignant' and 'Negative' class names are assigned red.)
- **Interactive Plotting**: Pan, zoom, and adjust color/transparency options.
- **Dynamic Visualization**: Reorder or invert axes to uncover patterns in n-D data.

## Data Interaction

- **Clipping Tools**: Dual-right click to highlight, analyze, export, or hide data.  
Clip Types, (all three are exported each clip analysis execution to CSV files):
  - Vertex clip: check if the vertex is inside the rectangle
  - Line clip: check if the line is inside the rectangle
  - End clip: check if the last vertex of the line is inside the rectangle
- **Point Selection**: Select points directly on the plot for detailed examination.

## Classification and Analysis

- **Rule-Based Classification**: Create and combine classification rules to build visual ML models.
- **Enhanced Learning Models**: Use interactive tools and synthetic data generation to improve supervised learning classifiers.

## Customization

- **Configurable Display**: Customize background, class, and axes colors.
- **Visibility Toggles**: Show or hide axes, points, and other elements.
- **Trace Mode**: Use alternating colors to trace data points across visualization schemes.

## User Interface Enhancements

- **Interactive UI Elements**: Manage data and visualization settings with intuitive controls.
- **Hotkeys and Shortcuts**: Access frequently used functions with keyboard shortcuts.

These features provide an interactive, user-friendly experience for data analysis.

## Mouse Interactions

- **Pan Plot**: Click and drag with the scroll-wheel.
- **Zoom**: Scroll the mouse wheel.
- **Box Clipping**: Right-click twice to create a clipping rectangle.
- **Select Point**: Left-click to select single or multiple points.
- **Grow Clipping Box**: Middle-click once to create clipping box, again to grow it.

## Keyboard Shortcuts

- **Cycle Selections**: `Q` and `E` keys.
- **Adjust Vertical Position**: `W` and `S` keys.
- **Delete Samples**: `D` key.
- **Print Samples**: `P` key.
- **Clone Samples**: `C` key.
- **Insert Sample**: `I` key.
- **Relabel Samples**: `R` key.
- **Generate Synthetic Data via CTGAN**: `G` key.

## UI Elements

- **Highlight Clipped Cases**: Clipped cases are highlighted; use 'Add Classification Rule' to convert to a new rule.
- **Rule Visibility**: Toggle visibility with the checkbox next to the rule.
- **Reorder Tables**: Drag and drop within class and attribute tables.
- **Adjust Transparency**: Use the slider below the attribute table.

## Datasets Included

| Dataset                                | Cases  | Features | Classes | File Name                      |
|----------------------------------------|--------|----------|---------|--------------------------------|
| Fisher Iris                            | 150    | 4        | 3       | fisher_iris.csv                |
| Breast Cancer Wisconsin (30 features)  | 569    | 30       | 2       | breast-cancer-wisconsin.csv    |
| Breast Cancer Wisconsin (9 features)   | 569    | 9        | 2       | breast-cancer-wisconsin-9f.csv |
| Diabetes                               | 768    | 8        | 2       | diabetes.csv                   |
| Heart Disease                          | 1,025  | 13       | 2       | heart_disease.csv              |
| Ionosphere                             | 351    | 34       | 2       | ionosphere.csv                 |
| MNIST Capital Letters                  | 20,000 | 16       | 26      | MNIST_letters.csv              |
| Sinusoidal Wave                        | 200    | 2        | 2       | sin_cos.csv                    |
| Wheat Seed                             | 210    | 7        | 3       | wheat_seeds.csv                |
| Wine                                   | 4,898  | 11       | 7       | wine.csv                       |
| Fisher Iris 2 (doubled classes)        | 300    | 4        | 6       | fisher_iris2.csv               |
| Fisher Iris Setosa vs Versicolor       | 100    | 4        | 2       | fisher_iris_SvVe.csv           |
| Iris Setosa                            | 50     | 4        | 1       | iris_setosa.csv                |
| Iris Setosa vs Virginica               | 100    | 4        | 2       | iris_S_vs_VV.csv               |
| Iris Versicolor vs Virginica           | 100    | 4        | 2       | iris_V_vs_V.csv                |
| Iris Setosa vs Versicolor vs Virginica | 150    | 4        | 3       | iris_SVe_vs_Vi.csv             |
| Synthetic Grades (100 cases)           | 100    | 3        | 3       | synthetic_grades_100case.csv   |
| Synthetic Grades (250 cases)           | 250    | 3        | 3       | synthetic_grades_250case.csv   |
| Synthetic Grades (25 cases)            | 25     | 3        | 3       | synthetic_grades_25case.csv    |
| Synthetic Grades (50 cases)            | 50     | 3        | 3       | synthetic_grades_50case.csv    |
| Synthetic Grades (75 cases)            | 75     | 3        | 3       | synthetic_grades_75case.csv    |
| MNIST Capital Letters A vs B           | 10,000 | 16       | 2       | MNIST_letters_AvB.csv          |
| Artificial Test                        | 10     | 2        | 2       | artiftest.txt                  |

 Datasets included meet data formatting requirements listed in the next section. Files placed in subfolders of the datasets folder are in the .gitignore to store personal data.

## Dataset Requirements

DCVis works with structured numerical datasets and requires a data format of:

- File Format: .txt or .csv
- Header Row: Must include headers for feature names and the label column.
- Label Column: Must be named 'class'.
- Feature Columns:
  - Variable number of columns
  - Labeled in the header
  - Must contain numerical data only

These requirements ensure proper data recognition and processing for accurate visualizations in DCVis. Datasets included meet these requirements.

## General Line Coordinates Visualizations

Generalized vertex class included for adding visualization methods.  
This visualization tool features multiple GLC visualisation methods:

- Parallel Coordinates (PC) ![PC](/screenshots/PC.png)
- Shifted Paired Coordinates (SPC) ![SPC](/screenshots/SPC.png)
- Dynamic Scaffold Coordinates 1 (DSC1) ![DSC1](/screenshots/DSC1.png)
- Dynamic Scaffold Coordinates 2 (DSC2) ![DSC2](/screenshots/DSC2.png)
- Static Circular Coordinates (SCC) ![SCC](/screenshots/SCC.png)
- Dynamic Circular Coordinates (DCC) ![DCC](/screenshots/DCC.png)

## Classification Examples

93.33% classification of Iris with three rules found in SPC
![Basic SPC classification](/screenshots/SPC_Iris_3_rules_93.png)

DCC Iris Setosa and Versicolor classes with LDA coefficients separation
![Basic DCC LDA separation](/screenshots/DCC_Setosa_Versicolor_LDA_separation.png)

## Publications

[1] Williams A., Kovalerchuk B., Representation Learning with Visual Knowledge Discovery, Human Computer Interaction International 2025 to be published as proceedings in Springer.

[2] Williams A., Kovalerchuk B., Synthetic Data Generation and Automated Multidimensional Data Labeling for AI/ML in General and Circular Coordinates, 2024 28th International Conference Information Visualisation (IV), pp.272-279, 2024, IEEE. Arxiv:2409.02079

[3] Recaido C., Kovalerchuk B., Interpretable Machine Learning for Self-Service High-Risk Decision-Making, in: 26th International Conference Information Visualisation, 2022, pp. 322–329, IEEE,  arXiv:2205.04032.

DCVis is a complete rebuild of the [DSCVis](https://github.com/Charles57-CWU/DSCVis) software, with new visualizations and enhanced tools which was built for [2].

DCVis is software developed at Central Washington University's Visual Knowledge Discovery and Imaging Lab. Initiated in 2022, the project is built by Alice Williams, James Battistoni IV, Charles Recaido, and led by Dr. Boris Kovalerchuk.

## License

DCVis is licensed under the MIT License, allowing for both personal and commercial use. For full license details, see the `LICENSE` file.
