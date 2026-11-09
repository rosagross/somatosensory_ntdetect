#!/bin/bash
#credits Burkhard Maess (maess@cbs.mpg.de)

called=$_
if [ -z "$BATDIR" ] ; then
    if [ $called != $0 ] ; then
        BATDIR=`dirname $BASH_SOURCE`                  # relative
        export BATDIR=`(builtin cd $BATDIR && pwd)`/   # absolutized and normalizes
    else
        BATDIR=`dirname $0`                            # relative
        export BATDIR=`(builtin cd $BATDIR && pwd )`/  # absolutized and normalized
    fi
    if [ -z "$BATDIR" ] ; then
        # error; for some reason, the path is not accessible
        # to the script (e.g. permissions re-evaled after suid)
        exit 1  # fail
    fi
fi
export EXPDIR=`(builtin cd $BATDIR/../.. && pwd )`/  # normalized
export EXPNAME=`basename $EXPDIR`
export EXPSUBJ=at
echo "EXPDIR:   " $EXPDIR " EXPNAME: " $EXPNAME " EXPSUBJ: " $EXPSUBJ
export WDDIR=${EXPDIR}grossmannr_wd/
echo "WDDIR:    " $WDDIR
export BATDIR=${WDDIR}bat/
echo "BATDIR:   " $BATDIR
export PYDIR=${WDDIR}code/
echo "PYDIR:    " $PYDIR
export RAWDIR=${EXPDIR}rawdir/
echo "RAWDIR:   " $RAWDIR
export DOCDIR=${WDDIR}doc/
echo "DOCDIR:   " $DOCDIR
export SUBJECTS_DIR=${EXPDIR}freesurfer/
echo "SUBJECTS_DIR:  " $SUBJECTS_DIR
export RESDIR=${EXPDIR}results_$USER/ #insert personal dir here
echo "RESDIR: " $RESDIR
export DATADIR=${EXPDIR}datadir/
echo "DATADIR: " $DATADIR
export PERSDATA=${WDDIR}data/
echo "DOCDIR:   " $DOCDIR

# I include this path because there are files that I need for the forward model computation
export PATH=$PATH:/afs/cbs.mpg.de/software/users/maess/scripts/

export MATLAB_VERSION=9.16
export MATLAB_REV=R2024a
export MATLAB_PREFDIR=${WDDIR}.matlab_${USER}/${MATLAB_REV}

MLPTEMP=${WDDIR}matlab
if [ -h $MLPTEMP ]
then 
   cd `readlink $MLPTEMP`
   export MATLABPATH=${MATLABPATH}:`pwd`
   cd -
else 
   export MATLABPATH=${MATLABPATH}:${WDDIR}matlab:${WDDIR}matlab/lib:${WDDIR}eeglab2024.2/
fi

echo " "
echo "Matlab activated - Version: ${MATLAB_VERSION} (${MATLAB_REV})"
echo " "

hoststr=`host $HOSTNAME`
hostip=`perl -e 'printf "%s",$ARGV[3];' $hoststr`
network=`perl -e '@a=split(/\./,$ARGV[0]); printf"%s.%s.%s",$a[0],$a[1],$a[2]' $hostip`
if [ $network == 192.168.60 ]
then
export LISTOFSUBJECTS=${CFGDIR}bw_list_of_subjects.lst
export WHEREAMI=MEGLAB
else
export LISTOFSUBJECTS=${CFGDIR}list_of_subjects.lst
export WHEREAMI=MPICBS
fi

if [ $WHEREAMI == MEGLAB ]
then
# export FIELDTRIPPATH=/nobackup/nigeria2/clozep/fieldtrip_20220125/
# export PYTHONPATH=${PYDIR}
t=1
else
# export FIELDTRIPPATH=/data/pt_00609/clozep/fieldtrip_20220125/
# export PYTHONPATH=${PYDIR}
t=2
fi
echo "FIELDTRIPPATH: " $FIELDTRIPPATH
echo "PYTHONPATH:    " $PYTHONPATH
export MNE_VERBOSE=0
echo "MNE_VERBOSE: $MNE_VERBOSE"

#MNEPYTHONVERSION=dev
#echo "Prepared Version of MNE-Python: ${MNEPYTHONVERSION:=v022}"
#export MNEPYTHONVERSION
#export MNE_DATASETS_SAMPLE_PATH=${EXPDIR}

export PATH=$PATH:$BATDIR
# python-MNE environment - activate python 3 enviroment with MNE-tools-python
if [ $WHEREAMI == MEGLAB ]
then
#eval "$(/nobackup/nigeria2/miniconda3/bin/conda shell.bash hook)" 
#. /nobackup/kuba2/miniconda3/etc/profile.d/conda.sh
#conda activate mne_${MNEPYTHONVERSION}
t=1
else
#source "/data/pt_00609/miniconda3/etc/profile.d/conda.sh"
#conda activate mne_${MNEPYTHONVERSION}
t=2
fi

umask 002
#newgrp neuro
source /afs/cbs.mpg.de/software/scripts/MATLAB -v $MATLAB_VERSION
if [ $0 == $called ]; then
exec bash
fi

