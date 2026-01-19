# Disruption measure: Time-Normalised PageRank

This directory contains the code to run the time-normalised PageRank disruption measure. 

## Structure

The src contains three main directories to run the measure:
- `async_loader`: Contains code to asynchronously load data from S3. The class is relatively generic and can be used to load any data from S3.
- `edge_list_loader`: Contains code to clean load the edge list data from S3. Also loads graph info data which is used in combination with the edge list to filter the edge list where appropriate.
- `graph_metrics`: Contains the code to run the PageRank algorithm on the edge list data. It also contains the code to normalise the PageRank scores by time.
- `pagerank_pipeline`: Contains the main pipeline code to run the PageRank algorithm and normalise the scores by time. It also contains the code to save the results to S3.

The code is designed to be modular, the async loader can be used to load any data from S3, while the edge list loader could be used for any sort of graph.

The graph metrics files meanwhile are set up for PageRank but other graph metrics could be added in the future.

## Dependencies
The code is designed to run in a conda environment. The dependencies are listed in the `environment.yaml` file.

Also included is a `pyproject.toml` file. This enables usage of this project as a package in other uv workspaces. To install to a different workspace in impact measures you will need to install them in editable mode:
```bash
uv pip install -e whole_portfolio/disruption_measure
```
Alternatively you can install directly from github using:
```
uv add git+https://github.com/wellcometrust/impact_measures.git#subdirectory=whole_portfolio/disruption_measure/rescaled_pagerank
```

**Running PageRank will not work without the conda environment**. This is because of graph-tool, a compiled C++ package which is not on pypi, only conda-forge. You can optionally download and install from system package managers such as brew or apt and manage dependencies yourself. All other code can be run without the conda environment and fully managed by uv.

## How to run 

First please be aware that running pagerank on every publication in the WAG is incredibly memory intensive. Please using a machine with around 200gbs of RAM should you wish to run it on everything.

This assumes you are in the root directory, running on an ec2 machine or similar.

First set up the environment using conda:

```bash
conda env create -f  whole_portfolio/disruption_measure/environment.yaml
conda activate gt
```
Then run PageRank including any args you may wish to alter from default:

```bash
python -m whole_portfolio.disruption_measure.rescaled_pagerank.pagerank_pipeline.pagerank_pipeline \
    --save-locally \
    --time-normalise \
    --test
```

In this example, I am running PageRank in test mode. I am using the --save-locally argument which saves the data after the initial processing step. Should I wish to alter any pagerank parameters after I can run again with the --resume-locally argument without having to reprocess the data.
