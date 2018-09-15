#!/bin/bash
########################################
## AIStack, v. 0.7-FPOS (15/09/2018)  ##
## Feature-packed | one-shot version  ##
########################################
#
# A deterministic, Anaconda-powered, PyTorch-based, AI research deployment
# installer, with a focus on deep learning, deep probabilistic programming and
# inference (Bayesian included), reinforcement learning and AI-powered
# science/experimentation. Now with some power-ups and a leaner install procedure.
#
# (c) 2018 Emanuele Ballarin <emanuele@ballarin.cc>
# Released under the GNU-LGPL v3.
#
#
# §§ REQUIREMENTS §§
#
# - Relatively recent Linux operating connected to the Internet (tested on Arch);
# - Bash v.4;
# - Anaconda Python distribution (i.e. Anaconda, MiniConda, IntelConda, ...);
# - An Intel processor (recommended: series 6 or better) with AVX2 support;
# - Intel MKL 2018 (or 2019 beta) full library, or Parallel Studio XE 2018/2019;
# - Intel MKL-DNN (recommended: self-compiled with Intel Parallel Studio XE);
# - NVidia CUDA v.9.0 or better;
# - NVidia CUDNN v.7.0 or better;
# - NVidia NCCL (whichever version compatible with CUDA);
# - Intel and NVidia "stuff" needs to be correctly sourced as per install guide(s);
# - OpenMPI v.3 (recommended: self-compiled with optimizations);
# - Kitware's CMAKE v.3.11 (if unavailable it needs to be source-compiled);
# - GNU Compiler Suite v.7 (C/C++/Fortran), with executables in $PATH;
# - GNU MultiPrecision Library (whicever version compatible);
# - GNU LibTool v.2.4 or better;
# - GNU AutoMake v.1.15 or better;
# - Approximately 2.5 hours of time (may take more with older CPUs);
#
# §§ HACKING §§
# Feel free to edit this script and experiment with it. However, it should be
# noted that the procedures involved partially push the standard Conda use-case
# at the edges (i.e. forced uninstalls, symlinking system deps inside Conda...).
# Expect some adventures if you substantially diverge from the already traced
# path. Don't be scared, though! :-)
#
# §§ FPOS: WHAT DOES IT MEAN? §§
# Trying to be as concise as possible, it means three things: (a) that this
# version has a lot of dependencies, which provide extensive functionality,
# (b) that Anaconda's dependency resolution is sometimes sub-optimal or very
# opaque in operation, (c) that in this specific case it goes in an endless
# loop if trying to call for a global update after the installation of all the
# packages. This is the reason it is not possible to do it anymore.
# This, however, should cause no problem whatsoever... at least as far as I know.
###


## USER-EXPOSED CONFIGURATION ##

# PyTorch version:
#export SELF_PTBRANCH_V="v0.4.1"                                         # The version of PyTorch you want to install ("master" for the latest unstable)
export SELF_PTBRANCH_V="master"

# Anaconda
export SELF_CEACT_COMMAND="intelactivate"                               # Command used to activate conda environments (usually "activate" as in "source activate ...")
export SELF_CONDA_ENV_NAME="aistack"                                    # Name of Conda environment to contain this setup
export SELF_CONDA_ENV_PATH="$HOME/intel/intelpython3/envs/"             # Path under which given command(s) will create Anaconda environments (must be manually specified due to possible multiple conda binaries installed!)

# CPU & Intel Accelerators
export SELF_NRTHREADS="8"                                               # Number of CPU threads to use in the building process
export SELF_INTELMKL="/opt/intel/compilers_and_libraries_2019.0.117"    # Path to Intel compilers and libraries directory, with explicit version reference
export SELF_MKLDNN_PATH="/usr/local/lib/"                               # External path of 'libmkldnn.so' such that $SELF_MKLDNN_PATH/libmkldnn.so is valid
                                                                        # If libmkldnn.so has not been built from source, $SELF_MKLDNN_PATH/libmklml.so must also be valid.

# GPU, CUDA & NVidia Accelerators
export SELF_CUDARCH="5.0"                                               # Nvidia CUDA compute capability (depends on GPU) in the form 'x.y' (with x, y numbers)
export SELF_CUDA_ROOT="/opt/cuda/"                                      # External path for the root directory of the CUDA Toolkit
export SELF_CUDNN_PATH="/opt/cuda/lib64/"                               # External path of 'libcudnn.so' such that $SELF_CUDNN_PATH/libcudnn.so is valid

# Distributed execution
export SELF_MPIROOT="/opt/openmpi/"                                             # Base directory of OpenMPI or other MPI implementation (must contain /bin, /lib, /include... and the such)

# Execution flags
export SELF_FIRSTRUN="1"                                                # Set to "1" if you want to install a new environment, to "0" in all other cases
export SELF_XGBOOST="1"                                                 # Set to "1" if you want XGBoost Python bindings to be installed (currently experiencing some problems, though!)


# Prepare local temporary directory
export SELF_INVOKEDIR="$(pwd)"
rm -R -f ./aistack-giftwrap
mkdir -p ./aistack-giftwrap
cd ./aistack-giftwrap
export SELF_BASEDIR="$(pwd)"


# Prompt the creation/update of Conda environments and PIP operations
echo ' '
echo "You will be prompted to perform operations on a Conda environment. Please, always accept."
read -p "Press [ENTER] to proceed..."
echo ' '


## ANACONDA ENVIRONMENT PREPARATION ON FIRST RUN ##

if [ "$SELF_FIRSTRUN" = "1" ]; then

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
    wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://ballarin.cc/mirrorsrv/aistack/dot-condarc
    mv ./dot-condarc ./.condarc

    # Setup symlinks
    cd $SELF_BASEDIR
    source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
    export SELF_CONDA_PREFORM="$(which conda)"
    source deactivate
    ln -s "$SELF_CONDA_PREFORM" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/conda"

    # Formalize Conda environment creation
    source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
    conda upgrade -y --all
    source deactivate

    # Install dependencies from Anaconda, as much as possible
    source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
    echo ' '
    conda install -y intelpython numpy scipy matplotlib sympy scikit-learn pyyaml typing six pandas networkx requests jpeg zlib tornado cython daal h5py hdf5 pillow pycparser isort ply jinja2 arrow singledispatch mypy mypy_extensions dask mkl-devel mkl-dnn mkl-include mkl mkl_fft mkl_random icc_rt tbb greenlet protobuf psutil intervals nose numba cryptography glib gmp icu idna flask libffi libgcc libgcc-ng libgfortran-ng libstdcxx-ng asn1crypto openssl pyopenssl openmp theano seaborn cffi future affine zeromq setuptools pip pydaal yaml pydot backports statsmodels llvmlite graphviz openpyxl certifi click cloudpickle execnet more-itertools mpmath numexpr rope simplegeneric sqlite tcl tk pcre pexpect ptyprocess py pytables python-dateutil keras-gpu==2.2.2 tensorflow-gpu==1.10 fastrlock filelock theano==1.1 pyzmq tqdm autograd scikit-image scikit-optimize jupyter jupyter_client jupyter_console jupyter_core jupyterlab jupyterlab_launcher notebook ipykernel ipyparallel ipython ipython_genutils ipywidgets ninja widgetsnbextension pytest pytest-runner websocket-client nbconvert nbformat nbsphinx nbstripout nbval sphinx sphinxcontrib sphinxcontrib-websupport sphinx_rtd_theme imageio imagehash ipdb numpydoc pytest-cov flake8 pytest-xdist pybind11 yapf pypandoc pep8-naming wheel virtualenv mock pytest-mock tox spacy tabulate attrs jedi typing-extensions pytest-runner recommonmark sphinx-autobuild sortedcontainers sortedcollections pycodestyle progressbar2 coveralls bumpversion scrapy coverage xarray docker-pycreds appdirs packaging pyparsing urllib3 pytest-timeout quantities ordered-set pyflakes libunwind autopep8 spyder-kernels cartopy astropy termcolor terminado pydotplus opencv markdown markupsafe livereload pyopengl httplib2 pathtools pylint pyqt jsonschema parso path.py patsy pickleshare qt terminado python-dateutil wrapt cytoolz dill eigen sparsehash jupyter_contrib_nbextensions bcolz feather-format plotnine msgpack-python keras-preprocessing keras-applications ansiwrap boto3 vcrpy requests metakernel cached-property apscheduler sqlalchemy alembic gevent peewee testfixtures pbr traitlets pytz django django-extensions faker pyscaffold dask-ml scikit-mdr skrebate ncurses cuda92 magma-cuda92 glfw3 docopt -c intel -c conda-forge -c pytorch -c menpo
    conda remove -y cmake cudatoolkit curl --force
    source deactivate
fi


# Fix problems
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
conda remove -y cmake cudatoolkit curl --force
source deactivate


# Fix the nasty cmake/ccmake bug
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
ln -s "$(which cmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
ln -s "$(which ccmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"


# Activate environment
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME


# If nonexistent, link MAGMA system-wide (it's hard to produce a *.a shared library correctly compiled with -fPIC) and make it discoverable
if [ ! -f /usr/local/lib/libmagma.a ]; then
    if [ ! -f /usr/lib/libmagma.a ]; then
        sudo ln -s "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/lib/libmagma.a" "/usr/local/lib/libmagma.a"
        sudo ldconfig -v
    fi
fi
if [ ! -f /usr/local/lib/libmagma_sparse.a ]; then
    if [ ! -f /usr/lib/libmagma_sparse.a ]; then
        sudo ln -s "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/lib/libmagma_sparse.a" "/usr/local/lib/libmagma_sparse.a"
        sudo ldconfig -v
    fi
fi


## PROGRAMMATICALLY EXPORT RELEVANT CONFIGURATION VARIABLES ##

# Distributed execution
export MPI_C_COMPILER=mpigcc
export MPI_CXX_COMPILER=mpigxx
export MPI_Fortran_COMPILER=mpifort
export MPI_FORTRAN_COMPILER=mpifort
export MPI_FC_COMPILER=mpifort
export PATH="$SELF_MPIROOT/bin/:$PATH"
export LD_LIBRARY_PATH="$SELF_MPIROOT/lib/:$SELF_MPIROOT/lib/openmpi/:$LD_LIBRARY_PATH"

# CPU & Intel Accelerators
export CFLAGS="-I$SELF_INTELMKL/linux/mkl/include/:$CFLAGS"
export CMAKE_LIBRARY_PATH="$SELF_INTELMKL/linux/mkl/lib/intel64/:$CMAKE_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="$MKL_INCLUDE:$CMAKE_INCLUDE_PATH"
export CMAKE_LIBRARY_PATH="$MKL_LIBRARY:$CMAKE_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="$CPATH:$CMAKE_INCLUDE_PATH"
export CMAKE_LIBRARY_PATH="$LIBRARY_PATH:$CMAKE_LIBRARY_PATH"
export MKLDNN_LIBRARY="$SELF_MKLDNN_PATH/libmkldnn.so"

# GPU, CUDA & NVidia Accelerators
export CUDA_TOOLKIT_ROOT_DIR="$SELF_CUDA_ROOT"  # Required by Tensor Comprehensions
export CUDNN_LIBRARY="$SELF_CUDNN_PATH/libcudnn.so"

# Compilers
export CC=gcc-7
export CXX=g++-7
export FC=gfortran-7
export CMAKE_C_COMPILER="$CC"
export CMAKE_CXX_COMPILER="$CXX"
export CMAKE_Fortran_COMPILER="$FC"
export CMAKE_FORTRAN_COMPILER="$FC"
export CMAKE_FC_COMPILER="$FC"
export cc="$CC"
export cxx="$CXX"
export fc="$FC"


## PRELIMINARY PIP DEPENDENCIES ##

cd $SELF_BASEDIR
rm -R -f ./pipdeps
mkdir -p ./pipdeps
cd ./pipdeps

# Meta
git clone --recursive https://github.com/srossross/Meta.git
cd Meta
pip install --upgrade --no-deps ./
cd ../

# Pyglet
pip install --upgrade --no-deps pyglet

# Observations
pip install --upgrade --no-deps observations

# Pillow-SIMD
pip uninstall -y pillow
pip uninstall -y pillow
pip uninstall -y pillow
export SELF_OLDCC="$CC"
CC="cc -mavx2" pip install --upgrade --no-deps --force-reinstall pillow-simd
CC="cc -mavx2" pip install --upgrade --no-deps --force-reinstall pillow-simd
export CC="$SELF_OLDCC"

# Visdom
git clone --recursive https://github.com/facebookresearch/visdom.git
cd visdom
python setup.py install
cd ../

# MuJoCo-Py (OpenAI)
pip install --upgrade --no-deps mujoco-py


## PREPARE, COMPILE, INSTALL PYTORCH ##

export SELF_CMAKE_PREFIX_PATH_OLD="$CMAKE_PREFIX_PATH"
export CMAKE_PREFIX_PATH="$(dirname $(which python))"

# CPU
export MAX_JOBS=$SELF_NRTHREADS
export NO_MKLDNN=0
export WITH_MKLDNN=1

# GPU
export NO_CUDA=0
export WITH_CUDA=1
export NO_NNPACK=0
export WITH_NNPACK=1
export NO_SYSTEM_NCCL=0
export WITH_SYSTEM_NCCL=1
export NO_CUDNN=0
export WITH_CUDNN=1
export USE_STATIC_CUDNN=0
export WITH_STATIC_CUDNN=0
export WITH_NCCL=1
export TORCH_CUDA_ARCH_LIST="$SELF_CUDARCH"

# Software stack
export WITH_NINJA=1
export WITH_NUMPY=1

# Distributed execution
export NO_DISTRIBUTED=1
export WITH_DISTRIBUTED=0
export WITH_DISTRIBUTED_MW=0
export WITH_GLOO_IBVERBS=0
export WITH_ROCM=0

# Configuration flags
export DEBUG=0
export USE_SYSTEM_EIGEN_INSTALL=0

# Prepare temporary directory for compilation
cd $SELF_BASEDIR
rm -R -f ./pthbuild
mkdir -p ./pthbuild
cd ./pthbuild

# Clone repository
git clone --recursive -b $SELF_PTBRANCH_V https://github.com/pytorch/pytorch
cd ./pytorch

# Uninstall previous version(s)
pip uninstall -y torch
pip uninstall -y torch
pip uninstall -y torch

# Build and install PyTorch
python setup.py install

# Restore default Cmake paths after install
export CMAKE_PREFIX_PATH="$SELF_CMAKE_PREFIX_PATH_OLD"


## PREPARE, COMPILE, INSTALL EXTRA PACKAGES ##
cd $SELF_BASEDIR
rm -R -f ./extras
mkdir -p ./extras
cd ./extras

# TorchVision
git clone --recursive https://github.com/pytorch/vision.git
cd ./vision
python setup.py install
cd ../

# TorchAudio
git clone --recursive https://github.com/pytorch/audio.git
cd ./audio
python setup.py install
cd ../

# TorchText
git clone --recursive https://github.com/pytorch/text.git
cd text
python setup.py install
cd ../

# TNT
git clone --recursive https://github.com/pytorch/tnt.git
cd tnt
python setup.py install
cd ../

# Ignite
git clone --recursive https://github.com/pytorch/ignite.git
cd ignite
python setup.py install
cd ../

# Torchfile
pip install --upgrade --no-deps torchfile

# Pyro (patched to avoid that PyPandoc failures make fail the whole install process)
git clone --recursive https://github.com/uber/pyro
cd pyro
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://github.com/emaballarin/pyro/commit/7d91c09efc78514b3edb2aa9daa04c53cbf6e88e.patch
git apply ./7d91c09efc78514b3edb2aa9daa04c53cbf6e88e.patch
rm ./7d91c09efc78514b3edb2aa9daa04c53cbf6e88e.patch
pip install --upgrade --no-deps .
cd ../

# ProbTorch
pip install --upgrade --no-deps git+https://github.com/probtorch/probtorch

# MPI4Py
git clone --recursive https://bitbucket.org/mpi4py/mpi4py.git
cd mpi4py
pip install --upgrade --no-deps .
cd ../

# Graphviz (let it be duplicate with Anaconda!)
pip install --upgrade --no-deps graphviz

# SKORCH
git clone https://github.com/dnouri/skorch.git
cd skorch
python setup.py install
cd ../

# SCOOP
git clone --recursive https://github.com/soravux/scoop.git
cd scoop
python setup.py install
cd ../

# DEAP
git clone --recursive https://github.com/DEAP/deap.git
cd deap
python setup.py install
cd ../

# ELI5
git clone --recursive https://github.com/TeamHG-Memex/eli5.git
cd eli5
python setup.py install
cd ../

# ONNX
pip install --upgrade --no-deps onnx

# Pyre
pip install --upgrade --no-deps pyre-check

# PyStan
pip install --upgrade --no-deps pystan

# Prophet
pip install --upgrade --no-deps fbprophet

# Traces
pip install --upgrade --no-deps traces

# Facebook eXecutable ARchives
pip install --upgrade --no-deps xar

# PLYara
pip install --upgrade --no-deps plyara

# Physiq-FlatBuffers
pip install --upgrade --no-deps physiq-flatbuffers

# Docker-Python
pip install --upgrade --no-deps docker

# Matrices
pip install --upgrade --no-deps matrices

# PyLaTeX
pip install --upgrade --no-deps pylatex

# Numpy-STL
pip install --upgrade --no-deps numpy-stl

# Pandas Summary
pip install --upgrade --no-deps pandas-summary

# SKLearn Pandas
pip install --upgrade --no-deps sklearn-pandas

# CARL, custom-merged version
git clone --recursive https://github.com/emaballarin/carl.git
cd carl
python setup.py install
cd ../

# TensorLy
git clone --recursive https://github.com/tensorly/tensorly
cd tensorly
python setup.py install
cd ../

# Facebook SparseConvNet
git clone --recursive https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet
rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so    # Port of script ./build.sh
python setup.py install
cd ../

# pyRTC (NVRTC)
pip install --upgrade --no-deps pynvrtc

# LieLearn
git clone --recursive https://github.com/AMLab-Amsterdam/lie_learn.git
cd lie_learn
python setup.py install
cd ../

# s2cnn (Spherical ConvNets)
git clone --recursive https://github.com/jonas-koehler/s2cnn.git
cd s2cnn
python setup.py install
cd ../

# Paysage
git clone --recursive https://github.com/drckf/paysage.git
cd paysage
rm ./requirements.txt   # Precautionary measure
python setup.py install
cd ../

# PyVarInf
git clone --recursive https://github.com/ctallec/pyvarinf.git
cd pyvarinf
python setup.py install
cd ../

# pytorch_fft (NOT COMPATIBLE ANYMORE!)
# pip install --upgrade --no-deps pytorch-fft

# TensorBoardX
pip install --upgrade --no-deps tensorboardX

# GPTorch
pip install --upgrade --no-deps git+https://github.com/cornellius-gp/gpytorch.git

# TorchFold
pip install --upgrade --no-deps torchfold

# PyTorchViz
pip install --upgrade --no-deps git+https://github.com/szagoruyko/pytorchviz

# PyToune
pip install --upgrade --no-deps git+https://github.com/GRAAL-Research/pytoune.git

# Atari Py
pip install --upgrade --no-deps atari-py

# box2d-py
pip install --upgrade --no-deps box2d-py

# PyGame
pip install --upgrade --no-deps pygame

# OpenAI Gym
pip install --upgrade --no-deps gym
pip install --upgrade --no-deps 'gym[all]'

# BindsNet
git clone --recursive https://github.com/Hananel-Hazan/bindsnet.git
cd bindsnet
pip install --upgrade --no-deps .
cd ../

# PLYFile
pip install --upgrade --no-deps plyfile

# PyTorch Scatter
git clone --recursive https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
python setup.py install
cd ../

# PyTorch Cluster
git clone --recursive https://github.com/rusty1s/pytorch_cluster.git
cd pytorch_cluster
python setup.py install
cd ../

# PyTorch Spline Cov
git clone --recursive https://github.com/rusty1s/pytorch_spline_conv.git
cd pytorch_spline_conv
python setup.py install
cd ../

# Pytorch Geometric
git clone --recursive https://github.com/rusty1s/pytorch_geometric.git
cd pytorch_geometric
python setup.py install
cd ../

# PyTorch BinCount (NOT COMPATIBLE ANYMORE!)
#git clone --recursive https://github.com/rusty1s/pytorch_bincount.git
#cd pytorch_bincount
#python setup.py install
#cd ../

# PyTorch Sparse
git clone --recursive https://github.com/rusty1s/pytorch_sparse.git
cd pytorch_sparse
python setup.py install
cd ../

# TorchBearer
git clone --recursive https://github.com/ecs-vlc/torchbearer.git
cd torchbearer
python setup.py install
cd ../

# PyCMA
git clone --recursive https://github.com/CMA-ES/pycma.git
cd pycma
python setup.py install
cd ../

# Lagom (requires PyTorch "master"), tweaked to avoid dependency re-download
git clone --recursive https://github.com/zuoxingdong/lagom.git
cd lagom
pip install --upgrade --no-deps ./
cd ../

# HyperLearn for PyTorch
git clone --recursive https://github.com/danielhanchen/hyperlearn.git
cd hyperlearn
python setup.py install
cd ../

# PyProb (K. Cranmer et al., 2018), dependency-stripped version
cd $SELF_BASEDIR
cd ./extras
pip install --upgrade --no-deps git+https://github.com/emaballarin/pyprob.git

export SELF_SUSPENDED_EXEC_PWD="$(pwd)"


#########################
######## CHAINER ########
#########################
# The Chainer 'subsystem'

## Core packages ##

mkdir -p $SELF_BASEDIR/chainerbuild
cd $SELF_BASEDIR/chainerbuild

# CuPy
pip install --upgrade --no-deps cupy-cuda92

# IDEEP
pip install --upgrade --no-deps ideep4py

# Chainer
pip install --upgrade --no-deps chainer

# ChainerMN
pip install --upgrade --no-deps chainermn

# ChainerCV
pip install --upgrade --no-deps chainercv

# ChainerRL
pip install --upgrade --no-deps chainerrl

# Chainer-ONNX
pip install --upgrade --no-deps onnx-chainer

# ChainerUI
pip install --upgrade --no-deps chainerui

## Extra packages ##

mkdir -p $SELF_BASEDIR/chainerextras
cd $SELF_BASEDIR/chainerextras

# ChainerEX
git clone --recursive https://github.com/corochann/chainerex.git
cd chainerex
python setup.py install
cd ../

# chainer_sklearn
git clone --recursive https://github.com/corochann/chainer_sklearn.git
cd chainer_sklearn
python setup.py install
cd ../

# Chainer Chemistry
git clone https://github.com/pfnet-research/chainer-chemistry.git
cd chainer-chemistry
pip install --upgrade --no-deps ./
cd ../

########## END ##########
#########################


cd "$SELF_SUSPENDED_EXEC_PWD"

# Lazy Imports
pip install --upgrade --no-deps lazy-import

# LazyData
pip install --upgrade --no-deps lazydata

# Data science & PyTorch utilities from Sebastian Raschka
pip install --upgrade --no-deps git+https://github.com/rasbt/mlxtend.git#egg=mlxtend
pip install --upgrade --no-deps git+https://github.com/rasbt/mytorch.git
pip install --upgrade --no-deps mputil
pip install --upgrade --no-deps git+https://github.com/rasbt/watermark#egg=watermark
pip install --upgrade --no-deps pyprind

# Nteract and Netflix OSS
pip install --upgrade --no-deps azure-datalake-store
pip install --upgrade --no-deps nteract_on_jupyter
pip install --upgrade --no-deps papermill

# Mimic Chainer CV functionality
git clone --recursive https://github.com/kuangliu/torchcv.git
export SELF_PYVRS="$(python -c 'import sys; print(sys.version[0])').$(python -c 'import sys; print(sys.version[2])')"
cp -R ./torchcv "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/lib/python$SELF_PYVRS/site-packages/"

# ULiège - Montefiore AI Tools
git clone --recursive https://github.com/montefiore-ai/pt_inspector.git
cd pt_inspector
python setup.py install
cd ../

git clone --recursive https://github.com/montefiore-ai/image_datasets.git
cd image_datasets
python setup.py install
cd ../

# Super-awesome Jupyter Notebook / Script interplay tool
pip install --upgrade --no-deps jupytext

# Keras-like interface to PyTorch
git clone --recursive https://github.com/abhaikollara/flare.git
cd flare
python setup.py install
cd ../

# Hypothesis (upstream + Simone Robutti's CSV extension)
pip install --upgrade --no-deps hypothesis
git clone --recursive https://github.com/chobeat/hypothesis-csv.git
cd hypothesis-csv
python setup.py install
cd ../

# PyBasicBayes
git clone --recursive https://github.com/mattjj/pybasicbayes.git
cd pybasicbayes
pip install --upgrade --no-deps ./
cd ../

# Some Reinforcement Learning "stuff" from Edouard Leurent (INRIA Lille)
git clone --recursive https://github.com/eleurent/highway-env.git
cd highway-env
pip install --upgrade --no-deps ./
cd ../

git clone --recursive https://github.com/eleurent/rl-agents.git
cd rl-agents
pip install --upgrade --no-deps ./
cd ../

git clone --recursive https://github.com/eleurent/obstacle-env.git
cd obstacle-env
pip install --upgrade --no-deps ./
cd ../


###############################################
######## AUTOMATIC FEATURE ENGINEERING ########
###############################################

# Prerequisites:
pip install --upgrade --no-deps update_checker
pip install --upgrade --no-deps stopit

# Install XGboost: START #
# Fixes
source deactivate
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
ln -s "$(which cmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
ln -s "$(which ccmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME
# Actual installation
if [ "$SELF_XGBOOST" = "1" ]; then
    read -p "Now, XGBoost will be configured, built and installed as a Python module. Press [ENTER] to continue..."
    cd $SELF_BASEDIR
    cd ./extras
    git clone --recursive -b release_0.80 https://github.com/dmlc/xgboost
    cd xgboost
    mkdir builddir
    cd builddir
    echo ' '
    echo "Ready to configure XGBoost (remember to suppress warnings: C/CXX FLAGS: -w)?"
    read -p "Press [ENTER] to proceed..."
    echo ' '
    ccmake ../
    echo ' '
    echo "Ready to build XGBoost?"
    read -p "Press [ENTER] to proceed..."
    echo ' '
    make -j12
    echo ' '
    echo "Ready to install XGBoost?"
    read -p "Press [ENTER] when you are ready to build Python package..."
    echo ' '
    cd ../
    cd python-package
    python setup.py install
    cd ../
    cd ../
    echo ' '
fi
# Install XGboost: STOP #

# TPOT
pip install --upgrade --no-deps tpot

########## END ##########
#########################

# Fix the nasty cmake/ccmake bug
source deactivate
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
ln -s "$(which cmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
ln -s "$(which ccmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME


# Uninstall useless PIP packages...
pip uninstall -y enum34
pip uninstall -y matplotlib
pip uninstall -y scipy
pip uninstall -y websocket-client
pip uninstall -y docker-pycreds
pip uninstall -y networkx

pip uninstall -y enum34
pip uninstall -y matplotlib
pip uninstall -y scipy
pip uninstall -y websocket-client
pip uninstall -y docker-pycreds
pip uninstall -y networkx

pip uninstall -y enum34
pip uninstall -y matplotlib
pip uninstall -y scipy
pip uninstall -y websocket-client
pip uninstall -y docker-pycreds
pip uninstall -y networkx

# ... and reinstall them with Conda.
conda install -y --force matplotlib scipy networkx websocket-client docker-pycreds -c intel -c conda-forge
conda remove -y cmake cudatoolkit curl --force


# Fix the nasty cmake/ccmake bug
source deactivate
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
rm -f "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
ln -s "$(which cmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/cmake"
ln -s "$(which ccmake)" "$SELF_CONDA_ENV_PATH/$SELF_CONDA_ENV_NAME/bin/ccmake"
source $SELF_CEACT_COMMAND $SELF_CONDA_ENV_NAME


# Download spaCy data
python -m spacy download en
python -m spacy download fr
python -m spacy download it
python -m spacy download xx
echo ' '


# Rename the temporary build directory
cd $SELF_BASEDIR
cd ../
mv ./aistack-giftwrap ./aistack-giftwrap-old
cd ./aistack-giftwrap-old
export SELF_BASEDIR="$(pwd)"


# Install and enable useful IPython/Jupyter extensions
echo "The lines between ### should contain at least a success. They may contain a failure; in that case, it's just fine."
echo "################################################################################"
ipcluster nbextension enable
sudo ipcluster nbextension enable
echo "################################################################################"
jupyter nbextension install --py ipyparallel
jupyter nbextension enable --py ipyparallel
jupyter serverextension enable --py ipyparallel
jupyter serverextension enable --py jupyterlab
jupyter nbextension enable nteract_on_jupyter
jupyter nbextension enable -py nteract_on_jupyter
jupyter serverextension enable nteract_on_jupyter
jupyter serverextension enable -py nteract_on_jupyter


# Success message and final recommendations
echo ' '
echo "SUCCESS!"
echo "If no error is displayed, you are ready to go. Enjoy!"
echo ' '
echo "After that, you will probably want to delete the entire temporary folder:"
echo "$SELF_BASEDIR"
echo "which has been left for eventual failed-install diagnostics."
echo ' '
read -p "Press [ENTER] to exit. Bye bye!"
echo ' '
