#!/bin/bash

## Preliminary ##
export SELF_INVOKEDIR_J=$(pwd)

## Config ##
export SELF_CONDA_ENV_NAME="jupyterconda"
export SELF_CEACT_COMMAND="activate"
export SELF_CONDA_ENV_PATH="/home/emaballarin/miniconda3/envs/"
export SELF_ENVURL="https://ballarin.cc/mirrorsrv/aistack/dot-jupycondarc"

## Preparation ##

# Fully erase (eventual) previous environment with same name
source $SELF_CEACT_COMMAND base
conda env remove -y -n $SELF_CONDA_ENV_NAME
conda env remove -y -n $SELF_CONDA_ENV_NAME
rm -R -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME"
rm -R -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME"
source deactivate && source deactivate

# Create fake environment
cd "$SELF_CONDA_ENV_PATH"
mkdir -p "$SELF_CONDA_ENV_NAME"
mkdir -p "$SELF_CONDA_ENV_NAME/bin"

# Setup channels
cd "$SELF_CONDA_ENV_NAME"
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait "$SELF_ENVURL"
mv ./dot-jupycondarc ./.condarc

# Setup symlinks
export SELF_INVOKEDIR_INTERLEAVED=$(pwd)
cd && cd
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
export SELF_CONDA_PREFORM="$(which conda)"
source deactivate && source deactivate
ln -s "$SELF_CONDA_PREFORM" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/conda"
cd "$SELF_INVOKEDIR_INTERLEAVED"

# Formalize Conda environment creation
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
conda upgrade -y --all
source deactivate && source deactivate

## Installation ##
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME

# From Conda
conda install ansiwrap boto3 alabaster appdirs asn1crypto atomicwrites attrs automat babel backcall blas bleach bokeh boost-cpp bottleneck bqplot breathe bzip2 ca-certificates cairo certifi cffi cftime chardet click click-plugins cligj cling-patches cloudpickle constantly coverage cppzmq cryptography cryptography-vectors cryptopp curl cycler cython cytoolz dask dbus decorator defusedxml descartes dill distributed docutils entrypoints expat fiona flexx fontconfig freetype freexl gdal geopandas geos geotiff gettext giflib glib gmp graphite2 graphviz gst-plugins-base gstreamer h5netcdf h5py harfbuzz hdf4 hdf5 heapdict html5lib hyperlink icu idna imagesize incremental ipykernel ipyleaflet ipyparallel ipython ipython_genutils ipywebrtc ipywidgets jedi jinja2 jpeg json-c jsonschema jupyter jupyter_client jupyter_console jupyter_contrib_core jupyter_contrib_nbextensions jupyter_core jupyter_highlight_selected_word jupyter_latex_envs jupyter_nbextensions_configurator jupyterlab jupyterlab_launcher kealib kiwisolver krb5 libdap4 libffi libgcc-ng libgdal libgfortran libgfortran-ng libiconv libkml libnetcdf libpng libpq libsodium libspatialindex libspatialite libssh2 libstdcxx-ng libtiff libtool libuuid libxcb libxml2 libxslt livereload locket lxml markdown markupsafe matplotlib metakernel mistune mizani mkl_fft mkl_random mock more-itertools msgpack-python munch nbconvert nbformat nbsphinx nbstripout nbval ncurses netcdf4 nlohmann_json notebook openblas openjpeg openssl packaging palettable pandas pandoc pandocfilters pango parso partd patsy pbr pcre pexpect pickleshare pip pixman plotnine pluggy poppler poppler-data postgresql proj4 prometheus_client prompt_toolkit psutil psycopg2 pthread-stubs ptyprocess pugixml py pyasn1 pyasn1-modules pycparser pygments pyhamcrest pyopenssl pyparsing pyproj pyqt pysal pysocks pytest python python-dateutil pytz pyyaml pyzmq qt qtconsole readline requests rtree scipy send2trash service_identity setuptools shapely simplegeneric sip six snowballstemmer sortedcontainers sphinx sphinx_rtd_theme sphinxcontrib-websupport spyder-kernels sqlalchemy sqlite statsmodels tblib terminado testfixtures testpath tk toolz tornado traitlets traittypes twisted urllib3 numpy wcwidth webencodings wheel widgetsnbextension xarray xerces-c xorg-kbproto xorg-libice xorg-libsm xorg-libx11 xorg-libxau xorg-libxdmcp xorg-libxext xorg-libxpm xorg-libxrender xorg-libxt xorg-renderproto xorg-xextproto xorg-xproto xproperty xtl xz yaml zeromq zict zlib zope.interface future tqdm botocore -c conda-forge
conda install libgcc-7 xeus xleaflet xplot xwebrtc xwidgets -c QuantStack
conda install intel-openmp libopenblas mkl numpy-base util-linux

# Update from Conda
conda upgrade --all
conda upgrade --all

# Fix install from Conda
pip uninstall -y enum34
pip uninstall -y enum34
pip uninstall -y enum34
conda remove -y cmake cudatoolkit curl --force
conda remove -y cmake cudatoolkit curl --force

# From Pip
pip install --upgrade --no-deps azure-datalake-store
pip install --upgrade --no-deps nteract_on_jupyter
pip install --upgrade --no-deps papermill
pip install --upgrade --no-deps ipysheet
pip install --upgrade --no-deps jupytext
pip install --upgrade --no-deps paperspace

# Activate extensions
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

## END SCRIPT ##
cd "$SELF_INVOKEDIR_J"
