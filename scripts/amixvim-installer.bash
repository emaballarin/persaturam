#!/bin/bash
#############################################
## AmixVim Installer, v. 0.3 (05/08/2018) ##
#############################################
#
# A no-frills Vim compiler/installer, powered by
# Amir Salihefendic's vimrc.
#
# (c) 2018 Amir Salihefendic
# (c) 2018 Emanuele Ballarin <emanuele@ballarin.cc>
# Released under the MIT license.
###

# Become location-aware
SELF_CURDIR=$(pwd)

# Create temporary directory
mkdir ./amixvim-configurator
cd ./amixvim-configurator
SELF_BASEDIR="$(pwd)"

# Get upstream sources
cd $SELF_BASEDIR
git clone https://github.com/vim/vim.git --recursive

# Attempt compilation of sources
cd ./vim/src
make distclean
./configure --enable-fail-if-missing --enable-luainterp=yes --with-lua-prefix=/usr/local --enable-mzschemeinterp --enable-perlinterp=yes --enable-pythoninterp=yes --enable-python3interp=yes --enable-tclinterp=yes --enable-rubyinterp=yes --with-ruby-command=/usr/bin/ruby --enable-cscope --enable-terminal --enable-autoservername --enable-multibyte --enable-fontset --enable-gui=auto --enable-gtk2-check --enable-gnome-check --enable-gtk3-check --enable-motif-check --enable-athena-check --enable-nextaw-check --enable-carbon-check --with-vim-name=vim --with-ex-name=ex --with-view-name=view --with-features=huge --with-modified-by=emaballarin --with-compiledby=emaballarin --with-luajit --with-python-command=/usr/bin/python2 --with-python3-command=/usr/bin/python --with-tclsh=/usr/bin/tclsh --with-x
make -j12
make -j12

# Install Vim
echo ' '
echo "Was the compilation successful?"
read -p "If so, press [ENTER] to install Vim system-wide!"
echo ' '
sudo make install
sudo make install

# Configure Vim acccording to AmixVim
if [ -d "$HOME/.vim_runtime/" ]; then
    cd $HOME/.vim_runtime/
    git pull --rebase
    git pull --rebase
else
    git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
    cd ~/.vim_runtime/
    sh ~/.vim_runtime/install_awesome_vimrc.sh
    git pull --rebase
    touch ./my_configs.vim
fi

# Cleanup
cd "$SELF_CURDIR"
rm -R -f ./amixvim-configurator

# Success!
echo ' '
echo "SUCCESS!"
echo ' '
