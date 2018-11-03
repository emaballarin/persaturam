#!/bin/zsh

##~~~~~~~~~~~~~~~~##
##~~ Manjaro-ck ~~##
##~~~~~~~~~~~~~~~~##
## A minimal-modifications port to Manjaro Linux of Con Kolivas'
## linux-ck kernel, based on the packages distributed by `graysky`
## through the Arch User Repository.
## It assumes that all relevant dependencies are already in place.
## Tweaked for ZSH, Intel Skylake and Nvidia proprietary drivers.
#####################################################################


##################
## Get packages ##
##################

# Export variables
export MANJAROCK_TMPDIR="$(pwd)/TMPDIR"
export MANJAROCK_MCKPKG="$(pwd)/MCKPKG"

# Prepare build structure
mkdir -p "$MANJAROCK_TMPDIR"
mkdir -p "$MANJAROCK_MCKPKG"

# Clean-up previous build
cd "$MANJAROCK_TMPDIR"
rm -R -f ./*

# Download package snapshots
cd "$MANJAROCK_TMPDIR"
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://aur.archlinux.org/cgit/aur.git/snapshot/acpi_call-ck.tar.gz
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://aur.archlinux.org/cgit/aur.git/snapshot/linux-ck.tar.gz
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://aur.archlinux.org/cgit/aur.git/snapshot/nvidia-ck.tar.gz
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://aur.archlinux.org/cgit/aur.git/snapshot/virtualbox-ck-modules.tar.gz

# Untar packages
tar xf ./acpi_call-ck.tar.gz
tar xf ./linux-ck.tar.gz
tar xf ./linux-ck.tar.gz
tar xf ./nvidia-ck.tar.gz
tar xf ./virtualbox-ck-modules.tar.gz


############################
## Autoconfigure packages ##
############################

# acpi_call-ck
## -> No patching/configuring needed as for now.

# linux-ck
cd "$MANJAROCK_TMPDIR/linux-ck"
sed -i "s/_subarch=.*/_subarch=22/g" ./PKGBUILD
sed -i "s/_NUMAdisable=y.*/_NUMAdisable=/g" ./PKGBUILD

# nvidia-ck
cd "$MANJAROCK_TMPDIR/nvidia-ck"
export MANJAROCK_NVIDIAVERSION="$(nvidia-smi | grep -o -P '.{0,0}Driver.{0,20}' | grep -Eo '[0-9]{1,4}' | sed ':a;N;$!ba;s/\n/./g')"
sed -i "s/pkgver=.*/pkgver=$MANJAROCK_NVIDIAVERSION/g" ./PKGBUILD
sed -i "s/sha256sums=('.*/sha256sums=('SKIP'/g" ./PKGBUILD

# virtualbox-ck-modules
cd "$MANJAROCK_TMPDIR/virtualbox-ck-modules"
sed -i "s/_kernel=\".*/_kernel=\"\"/g" ./PKGBUILD
sed -i "s/_extramodules=extramodules-ck-.*/_extramodules=extramodules-ck/g" ./PKGBUILD
sed -i "s/pkgbase=(virtualbox-ck-modules).*/pkgbase=virtualbox-ck-modules/g" ./PKGBUILD
sed -i "s/pkgname=(virtualbox-ck-host-modules).*/pkgname=virtualbox-ck-host-modules/g" ./PKGBUILD


####################
## Build packages ##
####################

cd "$MANJAROCK_TMPDIR/linux-ck"
makepkg -Csfi

cd "$MANJAROCK_TMPDIR/nvidia-ck"
makepkg -Csf

cd "$MANJAROCK_TMPDIR/acpi_call-ck"
makepkg -Csf

## virtualbox-ck-modules ##

# Install required MAKE-dependencies
trizen -S virtualbox-guest-dkms virtualbox-host-dkms

# Build package
cd "$MANJAROCK_TMPDIR/virtualbox-ck-modules"
makepkg -Csf

# Remove (now useless) MAKE-dependencies previously installed
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms


#####################
## Deploy packages ##
#####################

# Ask if deployment/install is really wanted
echo ' '
bash -c "read -p 'Was the whole build process successful? Press [ENTER] to deploy and install MANJARO-ck!'"
echo ' '

# Remove previously built packages
rm -R -f "$MANJAROCK_MCKPKG/*"


cd "$MANJAROCK_TMPDIR/linux-ck"
cp ./*.pkg.tar.xz "$MANJAROCK_MCKPKG"

cd "$MANJAROCK_TMPDIR/nvidia-ck"
cp ./*.pkg.tar.xz "$MANJAROCK_MCKPKG"

cd "$MANJAROCK_TMPDIR/acpi_call-ck"
cp ./*.pkg.tar.xz "$MANJAROCK_MCKPKG"

cd "$MANJAROCK_TMPDIR/virtualbox-ck-modules"
cp ./*.pkg.tar.xz "$MANJAROCK_MCKPKG"


######################
## Install packages ##
######################
cd "$MANJAROCK_MCKPKG"
sudo pacman -U ./*
