#!/bin/bash
ipcluster nbextension enable
sudo ipcluster nbextension enable
jupyter nbextension install --py ipyparallel
jupyter nbextension enable --py ipyparallel
jupyter serverextension enable --py ipyparallel
jupyter serverextension enable --py jupyterlab
jupyter nbextension enable nteract_on_jupyter
jupyter nbextension enable -py nteract_on_jupyter
jupyter serverextension enable nteract_on_jupyter
jupyter serverextension enable -py nteract_on_jupyter
