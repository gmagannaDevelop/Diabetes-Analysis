# Diabetes-Analysis

Main side project of [Gustavo Magaña López](https://github.com/gmagannaDevelop)

Contact me at [gmaganna.biomed@gmail.com](mailto:gmaganna.biomed@gmail.com).

## Disclaimer :
This is a personal project. It has two and only two scopes:

1. Help me understand better my diabetes, i.e. quench a personal and academic  curiosity.
2. Explore different ways of adjusting my therapy, finding patterns through diverse techniques and algorithms.

If you decide to test it on yourself it is **YOUR RESPONSIBILITY**. Nothing within this repo is enodorsed by Medtronic or MySugr App. I am merely a patient using his own data and applying algorithms to it. 


## Bibliography 
This project departed from my own knowledge of diabetes physiopathology. However, to make of this a valuable state-of-the-art tool I have decided to add also some docs. Publicly available scientific papers. These will be found in ```Docs/```. If you want to consult them, these are the original sources:

1. [review\_of\_formulas.pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4960276/)
2. [glycaemic-variability.pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5846572/pdf/dia.2017.0187.pdf)

If you have found anything that you consider pertinent/valuable for this project and have the proper copyright/ownership rights to share it, please do so! I'd be ravished to include it in the project ([email it to me](mailto:gmaganna.biomed@gmail.com), or add it to your fork of this repo and send me a pull request).


## Requirements and dependencies :
### Tested Hardware
This software has only been tested on a [13-inch, Mid 2012 MacBookPro](https://support.apple.com/en-us/HT201624) which is considered *vintage* (a.k.a. pre-obsolete). I've tampered with it, so it has 8 GiB RAM and a Samsung 1TB SSD.

I'm running macOS Mojave 10.14.6.


### Create a virtual environment using  `conda`
All of the code has been developed using conda. Using the provided files within the repo "**env.yml**" and "**requirements.txt**" will facilitate running the scripts and notebooks here present. For futher information consult:

1. [Anaconda distribution](https://www.anaconda.com/distribution/).
2. [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

To create the virtual environment, make sure you have installed the [latest version of Anaconda](https://conda.io/en/latest/).
Run the following command replacing ENV_NAME with the name you would like to give to the virtual environment:
  
 
     conda env create --name ENV_NAME --file env.yml
 
You will be prompted for confirmation, accept typing '**y**' on the interactive session.
To activate the newly created environment, type the following command on your terminal:

     conda activate ENV_NAME
     
Some dependencies (those which could not be installed through conda) were installed using `pip`. This is not the standard installation found on your machine (if you already had Python installed). Verify that you are using the correct `pip` by activating the virtual environment that you have designated for this repo (can be done via `which pip`).

Before installing dependencies via `pip`, **make sure you have activated the virtual environment running `conda activate ENV_NAME`** . Afterwards type:

    pip install -r requirements.txt 

 
Now you're ready to run the scripts and notebooks found on this repo. 

