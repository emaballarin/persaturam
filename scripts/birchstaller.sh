#!/usr/bin/zsh

# Become location-aware
export SELF_PWD_CALLDIR="$(pwd)"

# Create directories
rm -R -f ./birchstalldir
export BIRCHSTALLDIR="$SELF_PWD_CALLDIR/birchstalldir"
mkdir -p "$BIRCHSTALLDIR"

# Download sources
cd "$BIRCHSTALLDIR"
git clone https://github.com/lawmurray/Birch.git --recursive --branch gpu --single-branch --depth 1
git clone https://github.com/lawmurray/Birch.Standard.git --recursive --branch gpu --single-branch --depth 1
git clone https://github.com/lawmurray/Birch.Example.git --recursive --branch master --single-branch --depth 1
git clone https://github.com/lawmurray/Birch.SQLite.git --recursive --branch master --single-branch --depth 1
git clone https://github.com/lawmurray/Birch.Cairo.git --recursive --branch master --single-branch --depth 1

# Install the compiler
cd Birch
./autogen.sh
./configure
make -j8
sudo make install
cd ..

# Install additional libraries

cd Birch.Standard
birch build
sudo birch install
cd ..

cd Birch.Example
birch build
sudo birch install
cd ..

cd Birch.SQLite
birch build
sudo birch install
cd ..

cd Birch.Cairo
birch build
sudo birch install
cd ..


# Cleanup
cd "$SELF_PWD_CALLDIR"
rm -R -f "$BIRCHSTALLDIR"
