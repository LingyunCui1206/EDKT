
# Setup

To build and manipulate tree structures, which facilitates the visualization and analysis of hierarchical data, we first install the anytree package.
`pip install anytree`

Since the open-ended coding tasks we analyzed are in the Java language, the javalang package is installed to parse and process the Java codes, facilitating the extraction of the codes' syntactic structure and semantic information.
`pip install javalang`

# Processing

The path_extractor file is used to parse the path of the code and store it in labeled_paths.tsv.

The preprocessing file is used to split the training set, the validation set and the prediction set.


# Operation
run.py -> dataloader.py -> readdata.py
       -> evaluation.py -> EDKT.py
       
       
