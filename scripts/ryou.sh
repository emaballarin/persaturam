#!/usr/bin/env zsh

#######################################################
##            RYOU - Roll Your Own Unison            ##
## (C) 2021 Emanuele Ballarin <emanuele@ballarin.cc> ##
##             Distribution: MIT License             ##
#######################################################

################
### SETTINGS ###
################
SSH_USERNAME=" "
SSH_REMOTE=" "
SSH_DESTINATION=" "

################################################################################

# Become location-aware
export SELF_RYOU_CALLPATH="$(pwd)"

# Prepare
rm -R -f ./ryou
mkdir ./ryou
cd ./ryou

# Copy binaries
cp $(realpath $(which unison-text)) ./
cp $(realpath $(which unison-fsmonitor)) ./

# Link
ln -s ./unison-text ./unison

# Patch
patchelf --set-rpath "$SSH_DESTINATION" unison-text
patchelf --set-rpath "$SSH_DESTINATION" unison-fsmonitor
patchelf --set-interpreter "$SSH_DESTINATION"/ld-linux-x86-64.so.2 unison-text
patchelf --set-interpreter "$SSH_DESTINATION"/ld-linux-x86-64.so.2 unison-fsmonitor

# Copy libraries
cp /usr/lib/libutil.so.1 /usr/lib/libm.so.6 /usr/lib/libdl.so.2 /usr/lib/libc.so.6 /lib64/ld-linux-x86-64.so.2 ./

# Re-locate
cd "$SELF_RYOU_CALLPATH"

# Copy remotely
SSHSTRING="$SSH_USERNAME@$SSH_REMOTE:/$SSH_DESTINATION"
scp -rp ./ryou "$SSHSTRING"

# Cleanup
rm -R -f ./ryou

################################################################################
