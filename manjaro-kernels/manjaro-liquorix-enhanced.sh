#!/bin/zsh
# If running bash as the default shell, change the line above to: `#!/bin/bash` (without quotes).

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~ Manjaro-Liquorix, Enhanced ~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## A minimal-modifications port to Manjaro Linux of the Arch Linux flavour of the
## Liquorix Kernel (originally Debian-based), with additional extra-tweaking for
## improved performance and/or security.
##
## (c) 2018 Emanuele Ballarin <emanuele@ballarin.cc>
## Released to the public under the GNU-LGPL3 license.
##
## All due credit go the respective developers/maintainers, nominally:
## - The ZEN Kernel developers (maintainers: Steven Barrett, Jan Alexander Steffens, Miguel Botón)
## - The Liquorix kernel developers (maintainer: Steven Barrett)
## - GitHub user `graysky`
## - Piotr Górski
## - Joan Figueras
## - The Manjaro Linux developers (in particular Philip Mueller)
##
##
## § HOW THE PATCHING OCCUR?
## An Arch Linux-compatible source release of the Liquorix kernel is downloaded
## from the latest snapshot available from the AUR (maintainers: Steven Barrett,
## Piotr Górski). It is then adapted for installation and use in Manjaro Linux
## with specific substitutions in the PKGBUILD. The latest extramodules for the
## Manjaro kernel (maintainer: Philip Mueller) are fetched from their git repository,
## and adapted to Liquorix with specific substitutions in the PKGBUILD.
## Lastly, a .config-patching script (a modified version of the graysky-gcc-tuning
## script by Joan Figueras for the XanMod kernel) is made to be called with specific
## PKGBUILD substitution right before kernel building begins.
## This script applies to the kernel .config graysky-GCC-optimization-compatible
## flags (since source code modifications are already merged into Liquorix) and
## further tunes some extra parameters through specific substitution.
## All of the extramodules are instead left untouched.
##
## § DEPENDENCY NOTICE:
## It is assumed that all relevant dependencies are already in place when this
## script is called. Please, make sure this reflect your situation too.
##
## § OPTIMIZATION NOTICE:
## This script has been tweaked for an Intel Skylake processor and NVIDIA
## proprietary drivers.
############################################################################

##############
## TUNABLES ##
##############
LIQUORIX_CPU_ARCH="21"


##################
## Get packages ##
##################

# Export variables
export MANJAROLQX_TMPDIR="$(pwd)/TMPDIR"
export MANJAROLQX_LQXMPKG="$(pwd)/LQXMPKG"

# Prepare build structure
mkdir -p "$MANJAROLQX_TMPDIR"
mkdir -p "$MANJAROLQX_LQXMPKG"

# Clean-up previous build
cd "$MANJAROLQX_TMPDIR"
rm -R -f ./*

# Download package snapshots and clone relevant git repositories
cd "$MANJAROLQX_TMPDIR"
git clone --recursive https://aur.archlinux.org/linux-lqx.git
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/acpi_call.git
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/nvidia.git
git clone --recursive https://gitlab.manjaro.org/packages/extra/linux419-extramodules/virtualbox-modules.git

# Extract and put in place Joan Figueras' XanMod architecture selection script with custom patches
wget --tries=0 --retry-connrefused --continue --progress=bar --show-progress --timeout=30 --dns-timeout=30 --random-wait https://ballarin.cc/kernel-hacking/liquorix-reconf.sh
chmod +x ./liquorix-reconf.sh
cp ./liquorix-reconf.sh ./linux-lqx/liquorix-reconf.sh


############################
## Autoconfigure packages ##
############################

# linux-lqx
cd "$MANJAROLQX_TMPDIR/linux-lqx"
sed -i "s/_NUMAdisable=y.*/_NUMAdisable=/g" ./PKGBUILD
sed -i "/# Set these variables to ANYTHING.*/a _microarchitecture=$LIQUORIX_CPU_ARCH" ./PKGBUILD
sed -i "/	\[\[ -z \"\$_makegconfig\" \]\] || make gconfig/a 	\${srcdir}\/\.\.\/liquorix-reconf\.sh \$_microarchitecture" ./PKGBUILD

# acpi_call
cd "$MANJAROLQX_TMPDIR/acpi_call"
mv ./acpi_call.install ./acpi_call-liquorix.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-lqx/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-lqx/g" ./PKGBUILD
sed -i "s/install=\$_pkgname\.install.*/install=acpi_call-liquorix\.install/g" ./PKGBUILD
sed -i "s/acpi_call\.install\"/acpi_call-liquorix\.install\"/g" ./PKGBUILD

# nvidia
cd "$MANJAROLQX_TMPDIR/nvidia"
mv ./nvidia.install ./nvidia-liquorix.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-lqx/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-lqx/g" ./PKGBUILD
sed -i "s/install=nvidia\.install.*/install=nvidia-liquorix\.install/g" ./PKGBUILD
sed -i "s/nvidia\.install\"/nvidia-liquorix\.install\"/g" ./PKGBUILD

# nvidia -- remove if Liquorix version >= 4.19
sed -i -e '/# Linux 419 compat/,+2d' ./PKGBUILD

# virtualbox-modules
cd "$MANJAROLQX_TMPDIR/virtualbox-modules"
mv ./virtualbox-host-modules.install ./virtualbox-host-modules-liquorix.install
mv ./virtualbox-guest-modules.install ./virtualbox-guest-modules-liquorix.install
sed -i "s/_linuxprefix=.*/_linuxprefix=linux-lqx/g" ./PKGBUILD
sed -i "s/_extramodules=.*/_extramodules=extramodules-lqx/g" ./PKGBUILD
sed -i "s/install=virtualbox-host-modules\.install.*/install=virtualbox-host-modules-liquorix\.install/g" ./PKGBUILD
sed -i "s/install=virtualbox-guest-modules\.install.*/install=virtualbox-guest-modules-liquorix\.install/g" ./PKGBUILD
sed -i "s/virtualbox-host-modules\.install\"/virtualbox-host-modules-liquorix\.install\"/g" ./PKGBUILD
sed -i "s/virtualbox-guest-modules\.install\"/virtualbox-guest-modules-liquorix\.install\"/g" ./PKGBUILD
sed -i "s/package_linux419-virtualbox-host-modules/package_linux-lqx-virtualbox-host-modules/g" ./PKGBUILD
sed -i "s/package_linux419-virtualbox-guest-modules/package_linux-lqx-virtualbox-guest-modules/g" ./PKGBUILD


####################
## Build packages ##
####################

cd "$MANJAROLQX_TMPDIR/linux-lqx"
makepkg -Csfi

cd "$MANJAROLQX_TMPDIR/nvidia"
makepkg -Csf

cd "$MANJAROLQX_TMPDIR/acpi_call"
makepkg -Csf

## virtualbox-ck-modules ##

# Install required MAKE-dependencies
trizen -S virtualbox-guest-dkms virtualbox-host-dkms

# Build package
cd "$MANJAROLQX_TMPDIR/virtualbox-modules"
makepkg -Csf

# Remove (now useless) MAKE-dependencies previously installed
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms
sudo pacman -R virtualbox-guest-dkms virtualbox-host-dkms


#####################
## Deploy packages ##
#####################

# Ask if deployment/install is really wanted
echo ' '
bash -c "read -p 'Was the whole build process successful? Press [ENTER] to deploy and install MANJARO-Liquorix!'"
echo ' '

# Remove previously built packages
rm -R -f "$MANJAROLQX_LQXMPKG/*"


cd "$MANJAROLQX_TMPDIR/linux-lqx"
cp ./*.pkg.tar.xz "$MANJAROLQX_LQXMPKG"

cd "$MANJAROLQX_TMPDIR/nvidia"
cp ./*.pkg.tar.xz "$MANJAROLQX_LQXMPKG"

cd "$MANJAROLQX_TMPDIR/acpi_call"
cp ./*.pkg.tar.xz "$MANJAROLQX_LQXMPKG"

cd "$MANJAROLQX_TMPDIR/virtualbox-modules"
cp ./*host*.pkg.tar.xz "$MANJAROLQX_LQXMPKG"


######################
## Install packages ##
######################
cd "$MANJAROLQX_LQXMPKG"
sudo pacman -U ./*
