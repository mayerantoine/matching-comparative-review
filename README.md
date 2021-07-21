# matching-comparative-review

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mayerantoine/matching-comparative-review/master)

This repository allows to reproduce the experimental study comparing the effectiveness of 7 matching algorithms selected from the most used matching techniques like deterministic, probabilistic, and machine learning techniques. To conduct the experiment, we started by generating synthetic data from real-world names using the Freely Extensible Biomedical Record Linkage (FEBRL) software. Then we ran multiple deduplication algorithms on the synthetic data using the Python Record Linkage Toolkit (PRLT). Finally, we evaluated the effectiveness of the deduplication using matching quality metrics like recall, precision, and F score using PRLT.

To use or test this code no need install or setup python you can click on the "launch binder" button above.

## Using BinderHub

After clicking on the "launch binder" link above, wait for a few minutes to BinderHub build the Docker container.

## Run Locally

To run locally on your computer:

* Install Anaconda and jupyter notebooks or jupyter lab on your computer
* Clone or  Download the folder
* Install dependencies :  ```pip install requirements.txt```
* Open the jupyter notebook ```A comparative review of patient matching approaches.ipynb``` and run it.

## Folder contents

This folder contains :

* The main jupyter notebook :  ```A comparative review of patient matching approaches.ipynb```
* a Python module ``` patientlinkr.py``` required to run the notebook
* The datasets results when you run the notebooks
* A dataset folder with the 3 datasets to deduplicate
* A docs folder with images, a README file and requirements file

## Credits and Acknowledgements

The original first names and last names used to generate the synthetic datasets were scraped from :

* https://africa-facts.org
* https://answersafrica.com
* https://www.behindthename.com
* http://www.americanlastnames.us

