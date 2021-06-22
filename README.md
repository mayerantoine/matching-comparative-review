# matching-comparative-review

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mayerantoine/matching-comparative-review/master)

This repository allows to reproduce the experimental study comparing the effectiveness of 7 matching algorithms selected from the most used matching techniques like deterministic, probabilistic, and machine learning techniques. To conduct the experiment, we started by generating synthetic data from real-world names using the Freely Extensible Biomedical Record Linkage (FEBRL) software. Then we ran multiple deduplication algorithms on the synthetic data using the Python Record Linkage Toolkit (PRLT). Finally, we evaluated the effectiveness of the deduplication using matching quality metrics like recall, precision, and F score using PRLT.

To use or test this code no need install or setup python you can click on the "launch binder" link.

### Using BinderHub
After clicking on the "launch binder" link above, wait for a few minutes to BinderHub build the Docker container.