# laminar_uecog_viz
visualization functions for auditory experiments

notebooks folder: contains jupyter notebooks with early visialization functions 

    - Plot_Mean Inputs: TDT file with auditory data Outputs: Plot of the trialized mean response
    - Z-scoring_Saving_update.ipynb Inputs: TDT file with auditory data Outputs: Plot of the trialized Z-scored response
    - spect_helper_notebook_May_21_2021.ipynb Inputs: TDT file with auditory data Outputs: spectrogram (need more details on this)!

### From source
To install, you can clone the repository and `cd` into the process_nwb folder.

```bash
# use ssh
$ git clone 
$ cd process_nwb
```

If you are installing into an active conda environment, you can run

```bash
$ conda env update --file environment.yml
$ pip install -e .
```