
#!/bin/bash
USAGE=`cat << EOU

usage : $0 [datadir]
 
datadir	- name of the directory containing the data
          i.e. probands or testing
EOU`
if [ $# -lt 1 ] 
then
   echo "$USAGE" 
   echo " "
   exit 1
fi

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
export DDIR=`(builtin cd $BATDIR/../.. && pwd )`/  # normalized
export EXPNAME=NearThresholdDetection
export EXPSUBJ=bm
export TEMP=`(builtin cd $BATDIR/.. && pwd )`/  # normalized

export EXPDIR=${DDIR}$1'/'
if [[ ! -d $EXPDIR ]]
then
   echo "Error: This directory does not exist: $EXPDIR"
   echo "$USAGE" 
   echo " "
   exit 1
fi
echo "EXPDIR:   " $EXPDIR

echo "BATDIR:   " $BATDIR
export CFGDIR=${TEMP}cfg/
echo "CFGDIR:   " $CFGDIR
export CFGVPDIR=${TEMP}cfg_vp/
echo "CFGVPDIR: " $CFGVPDIR
export COREGDIR=${TEMP}coreg
echo "COREGDIR: ${COREGDIR}"
export COREGSOFTPATH=${DDIR}coregistration/head2mri
echo "COREGSOFTPATH: $COREGSOFTPATH"
export DOCDIR=${EXPDIR}doc/
echo "DOCDIR:   " $DOCDIR
if [[ ! ${DOCDIR} ]]
then 
   mkdir ${DOCDIR}
fi
export PYDIR=${TEMP}py/:${TEMP}py/lib/
#echo "PYDIR:     " $PYDIR
export PYTHONPATH=${PYDIR}

export RAWDIR=${DDIR}rawdir/
echo "RAWDIR:   " $RAWDIR

export LOCKDIR=${EXPDIR}lock/
echo "LOCKDIR:   " $LOCKDIR
if [[ ! ${LOCKDIR} ]]
then 
   mkdir ${LOCKDIR}
fi
export SUBJECTS_DIR=${RAWDIR}freesurfer/
echo "SUBJECTS_DIR:   " $SUBJECTS_DIR

USER=$(whoami)
export USER

hoststr=`host $HOSTNAME`
if [ hoststr == ltgrmeg4 ]
then
export WHEREAMI=ltgrmeg
export THISHOST=$HOSTNAME
else
hostip=`perl -e 'printf "%s",$ARGV[3];' $hoststr`
network=`perl -e '@a=split(/\./,$ARGV[0]); printf"%s.%s.%s",$a[0],$a[1],$a[2]' $hostip`
export THISHOST=$HOSTNAME
if [ $network == 192.168.60 ]
then
export LISTOFSUBJECTS=${CFGDIR}bw_list_of_subjects.lst
export WHEREAMI=MEGLAB
else
export LISTOFSUBJECTS=${CFGDIR}${1}_list_of_subjects.lst
export WHEREAMI=MPICBS
fi
fi
#export WHEREAMI=MPICBS
echo "Where am I: $WHEREAMI"

if [ $WHEREAMI == ltgrmeg ]
then
export PYTHONPATH=${PYDIR}
elif [ $WHEREAMI == EEGLAB ]
then
#export FIELDTRIPPATH=/nobackup/nigeria2/megdata/fieldtrip_2022XXXX/
export PYTHONPATH=${PYDIR}:${COREGDIR}
else
#export FIELDTRIPPATH=/data/pt_02659/fieldtrip_2022XXXX/
export PYTHONPATH=${PYDIR}
fi
#echo "FIELDTRIPPATH: " $FIELDTRIPPATH
echo "PYTHONPATH:    " $PYTHONPATH
export MNE_VERBOSE=0
echo "MNE_VERBOSE: $MNE_VERBOSE"

echo "Prepared Version of MNE-Python: ${MNEPYTHONVERSION:=v170}"
export MNEPYTHONVERSION
export MNE_DATASETS_SAMPLE_PATH=${EXPDIR}

export PATH=$PATH:$BATDIR:$COREGDIR:$COREGSOFTPATH
# python-MNE environment - activate python 3 enviroment with MNE-tools-python
if [ $WHEREAMI == ltgrmeg ]
then
eval "$(/opt/miniconda3/bin/conda shell.bash hook)" 
conda activate mne_${MNEPYTHONVERSION}
elif [ $WHEREAMI == MEGLAB ]
then
#eval "$(/nobackup/kalifornien2/miniconda3/bin/conda shell.bash hook)" 
eval "$(/nobackup/kansas2/practicalMEG/miniconda3/bin/conda shell.bash hook)" #?
conda activate mne_${MNEPYTHONVERSION}
else
eval "$(/data/pt_02934/miniconda3/bin/conda shell.bash hook)" #?
conda activate mne_${MNEPYTHONVERSION}
fi

#export MATLABPATH=${MATLABPATH}:${TEMP}matlab::${TEMP}matlab/lib
umask 002
newgrp users

if [ $0 == $called ]; then
exec bash
fi
