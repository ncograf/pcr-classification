# DataScienceLab
The folder Data is not tracked by Git and should contain
a copy of the Data folder available at [Data Shared by Miss McLeod](https://drive.switch.ch/index.php/s/Dk9O9BXu0drKvqU).
Note that this only matters for the default paths in the libraries, if you specify the corresponding
paths by yourself, you can place the files wherever you like.

## code
All of the code is in the `code` folder. 
The folder itself, contains a few jupyter notebooks, which document early experiments.
More detailed experiments were later moved to the corresponding folders:
- `custom_tkinter` contains the code for the beautiful gui
- `decorrelation` contains experiments and documentation of decorrelation (note that decorrelation turned out only to work with a lot of data and only if also a lot of positiv data was in the dataset. Hence decorelation was not used in the final algorithm.)
- `hierarchy` contains experiments (or rather a jupyter notebook set up to conduct experiments) with the hirearchy based algorithms
- `whitness_based` contains experiments (or rather a jupyter notebook set up to conduct experiments) with the witness based algorithms (we are aware of the misspelling, appologies for these mistakes.)
- `zero_cluster_detection` this folder contains experment files for the negative axis / control detection
- `libs` contains all the code which acutally does some computations. See below.

### libs
- `decision_lib` contains all the different algorithm we implemented and tested for point labelling. (The only ones which are use in the final version of the GUI are `cluster_relative_mean_classifier.py` and `whitnes_density_classifier.py` all others were sorted out due to worse performance or pore generalizabilty)
- `data_lib.py` contains code to load and prepare data. This library heavily depends on the structure (hierarchy) of the files and folders in the `Data` folder.
- `negative_dimension_detection.py` Code for detecting negative clusters. (There are multiple algorithms of which only one was used in the GUI)
- `plot_lib.py` everything related to plotting (check it out ;-) its fun)
- `stats_lib.py` computing concentation and statistics after the labelling is done or / and probabilites are computed
- `transform_lib` basically only related to whitening / decorrelation (also contains the cluster class (every library has some bad desing choices))
- `validation_lib` algorithms we used to compare our labels with the "true" labels provided by the challance givers.

## misc
Some of the results we exported to this folder but it's by far not complete due to a lack of self discipline.

## Data
Intended for data but not added to git due to the amount of data.

