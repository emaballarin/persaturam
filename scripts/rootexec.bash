#!/bin/bash
#
# rootexec (rootexec.bash)
# version: 0.2
rootexecversion="0.2"
#
# (C) Emanuele Ballarin <emanuele@ballarin.cc>, 2018-03-02
# Licensed under the terms of the CeCILL-C free/libre software license.
# Full text of the license: <https://pastebin.com/raw/TwstvSBK> or <http://www.cecill.info>
#
# rootexec is a (relatively) high-performance, pure-bash, wrapper and command
# parsing script for CERN's ROOT/Cling which enforces a juxtaposed, space-separated
# argument passing syntax typical of most popular scripting languages.
# It is compatible with Linux kernel's integrated binfmt_bin capabilities and
# provides basic error-handling features.
# The supported syntax targets any possible argument-passing supported by native
# ROOT/Cling and parses commands on a one-rule-only left-to-right basis.
# It is possible to force "pass as file" and "pass as argument" instructions
# with dedicated -e (execute) and -a (argument) modifiers. Modifiers are mandatory
# in case "-a" or "-e" must be passed as filepaths or arguments themselves.
# The provided command is first parsed and checked, then translated to a valid
# ROOT/Cling command, and only lastly executed.

exeextension="C"    # Edit to change the dafault C/C++ executable extension for ROOT/Cling

# Check if a valid interpreter is installed and configured correctly in the system.
# Defaults to ROOT. If not, defaults to Cling.
interp="$(which root)"
if [[ "$interp" = "" ]]; then
    interp="$(which cling)"
fi

# Respond to informational requests (--version)
if [[ "$1" = "--version" ]]; then
    echo "rootexec v. $rootexecversion"
    echo "Using $interp as interpreter and .$exeextension as default scriptable C/C++ extension"
    echo ' '
    exit
fi


if [[ "$interp" = "" ]]; then
    echo 'ERROR: No valid path for a compatible interpreter found. Check that you have ROOT or Cling installed and correctly sourced.'
    exit
fi

# Check for unequivocally undefined behaviour: argument without file
if [[ "$1" = "-a" ]]; then
    echo 'ERROR: You cannot pass arguments without specifying first the file you want to pass them to.'
    exit
fi

# Check for unequivocally undefined behaviour: trying to forcefully execute no-file
if [[ "${@: -1}" = "-e" ]]; then
    if [[ "${@:(-2):1}" != "-e" ]] && [[ "${@:(-2):1}" != "-a" ]]; then
        echo 'ERROR: You are trying to execute no-file.'
        exit
    fi
fi

# Initialize booleans to defaults
forceexec=false
forcearg=false
filelock=false

for arg in "$@"
do
    if [[ $forceexec = true ]]; then
        if [[ -f "$arg" ]]; then
            if [[ "$execfile" = "" ]]; then
                execfile="$arg"
                forceexec=false
                arglist=""
            else
                arglist="${arglist:1}"
                execandargs="$execfile($arglist)"
                interpcommand="$interpcommand $execandargs"
                execfile="$arg"
                forceexec=false
                arglist=""
            fi
        else
            echo 'ERROR: The file you want to execute does not exist.'
            exit
        fi

    elif [[ $forcearg = true ]]; then
        arglist="$arglist,$arg"
        forcearg=false

    elif [[ "$arg" = "-e" ]]; then
        forceexec=true

    elif [[ "$arg" = "-a" ]]; then
        forcearg=true

    elif [[ -f "$arg" ]] && [[ "${arg: -2}" == ".$exeextension" ]]; then
        if [[ "$execfile" = "" ]]; then
            execfile="$arg"
            arglist=""
        else
            arglist="${arglist:1}"
            execandargs="$execfile($arglist)"
            interpcommand="$interpcommand $execandargs"
            execfile="$arg"
            arglist=""
        fi

    else
        if [[ "$execfile" != "" ]]; then
            arglist="$arglist,$arg"
        else
            echo 'ERROR: No valid C/C++ executable file found, or trying to pass arguments to no-file.'
            exit
        fi
    fi

done

if [[ "$execfile" != "" ]]; then
    arglist="${arglist:1}"
    execandargs="$execfile($arglist)"
    interpcommand="$interpcommand $execandargs"
    execfile="$arg"
    forceexec=false
    arglist=""
fi

interpcommand="${interpcommand:1}"
"$interp" -l -x -q "$interpcommand"

echo ' '
