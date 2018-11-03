#!/bin/zsh

##~~~~~~~~~~~~~~~~~~~~##
##~~ Manjaro-XanMod ~~##
##~~~~~~~~~~~~~~~~~~~~##
## A minimal-modifications port to Manjaro Linux of Alexandre Frade's
## XanMod kernel, based on the packages distributed by Joan Figueras
## through the Arch User Repository and the official Philip Mueller's
## Manjaro kernel extramodules.
## It assumes that all relevant dependencies are already in place.
## Tweaked for ZSH, Intel Skylake and Nvidia proprietary drivers.
#######################################################################


##################
## Get packages ##
##################

# Export variables
export MANJAROXM_TMPDIR="$(pwd)/TMPDIR"
export MANJAROXM_MXMPKG="$(pwd)/MXMPKG"

# Prepare build structure
mkdir -p "$MANJAROXM_TMPDIR"
mkdir -p "$MANJAROXM_MXMPKG"

# Clean-up previous build
cd "$MANJAROXM_TMPDIR"
rm -R -f ./*

# Download package snapshots and clone relevant git repositories
cd "$MANJAROXM_TMPDIR"
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://aur.archlinux.org/cgit/aur.git/snapshot/linux-xanmod.tar.gz
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/acpi_call.git
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/nvidia.git
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/virtualbox-modules.git

# Untar packages
tar xf ./linux-xanmod.tar.gz
# All the other packages are git repositories and are already (uncompressed) folders.


############################
## Autoconfigure packages ##
############################

# linux-xanmod
cd "$MANJAROXM_TMPDIR/linux-xanmod"
sed -i "s/  _configuration=1.*/  _configuration=1/g" ./PKGBUILD
sed -i "s/  _microarchitecture=0.*/  _microarchitecture=21/g" ./PKGBUILD

# acpi_call
cd "$MANJAROXM_TMPDIR/acpi_call"
mv ./acpi_call.install ./acpi_call-xanmod.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-xanmod/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-4.19-xanmod/g" ./PKGBUILD
sed -i "s/install=\$_pkgname\.install.*/install=acpi_call-xanmod\.install/g" ./PKGBUILD
sed -i "s/acpi_call\.install\"/acpi_call-xanmod\.install\"/g" ./PKGBUILD


# nvidia
cd "$MANJAROXM_TMPDIR/nvidia"
mv ./nvidia.install ./nvidia-xanmod.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-xanmod/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-4.19-xanmod/g" ./PKGBUILD
sed -i "s/install=nvidia\.install.*/install=nvidia-xanmod\.install/g" ./PKGBUILD
sed -i "s/nvidia\.install\"/nvidia-xanmod\.install\"/g" ./PKGBUILD

# virtualbox-modules
cd "$MANJAROXM_TMPDIR/virtualbox-modules"
mv ./virtualbox-host-modules.install ./virtualbox-host-modules-xanmod.install
mv ./virtualbox-guest-modules.install ./virtualbox-guest-modules-xanmod.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-xanmod/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-4.19-xanmod/g" ./PKGBUILD
sed -i "s/install=virtualbox-host-modules\.install.*/install=virtualbox-host-modules-xanmod\.install/g" ./PKGBUILD
sed -i "s/install=virtualbox-guest-modules\.install.*/install=virtualbox-guest-modules-xanmod\.install/g" ./PKGBUILD
sed -i "s/virtualbox-host-modules\.install\"/virtualbox-host-modules-xanmod\.install\"/g" ./PKGBUILD
sed -i "s/virtualbox-guest-modules\.install\"/virtualbox-guest-modules-xanmod\.install\"/g" ./PKGBUILD
sed -i "s/package_linux419-virtualbox-host-modules/package_linux-xanmod-virtualbox-host-modules/g" ./PKGBUILD
sed -i "s/package_linux419-virtualbox-guest-modules/package_linux-xanmod-virtualbox-guest-modules/g" ./PKGBUILD


####################
## Build packages ##
####################

cd "$MANJAROXM_TMPDIR/linux-xanmod"
makepkg -Csfi

cd "$MANJAROXM_TMPDIR/nvidia"
makepkg -Csf

cd "$MANJAROXM_TMPDIR/acpi_call"
makepkg -Csf

## virtualbox-ck-modules ##

# Install required MAKE-dependencies
trizen -S virtualbox-guest-dkms virtualbox-host-dkms

# Build package
cd "$MANJAROXM_TMPDIR/virtualbox-modules"
makepkg -Csf

# Remove (now useless) MAKE-dependencies previously installed
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms


#####################
## Deploy packages ##
#####################

# Ask if deployment/install is really wanted
echo ' '
bash -c "read -p 'Was the whole build process successful? Press [ENTER] to deploy and install MANJARO-XanMod!'"
echo ' '

# Remove previously built packages
rm -R -f "$MANJAROXM_MXMPKG/*"


cd "$MANJAROXM_TMPDIR/linux-xanmod"
cp ./*.pkg.tar.xz "$MANJAROXM_MXMPKG"

cd "$MANJAROXM_TMPDIR/nvidia"
cp ./*.pkg.tar.xz "$MANJAROXM_MXMPKG"

cd "$MANJAROXM_TMPDIR/acpi_call"
cp ./*.pkg.tar.xz "$MANJAROXM_MXMPKG"

cd "$MANJAROXM_TMPDIR/virtualbox-modules"
cp ./*host*.pkg.tar.xz "$MANJAROXM_MXMPKG"


######################
## Install packages ##
######################
cd "$MANJAROXM_MXMPKG"
sudo pacman -U ./*
