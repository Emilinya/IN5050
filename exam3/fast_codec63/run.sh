#!/bin/bash
set -e

#
# USAGE: ./run.sh [--tegra hostname]
#

TEGRA_CMD="c63server"
TEGRA_ARGS="/opt/Media/foreman.yuv -o output.c63 -w 352 -h 288 -f 10"

PC_CMD="c63enc"
PC_ARGS=$TEGRA_ARGS

SRC_DIR="$(realpath $(dirname $0))"
DATE=$(date -u +%Y%m%d-%H%M%S)
RSYNC_ARGS="-rt --exclude=logs/ --exclude=.*"
BUILD_DIR="in5050-codec63-build"

RUNS=1

trap quit INT

function quit()
{
    echo
    echo "### Cleaning up ###"
    ssh $TEGRA "pkill -u \$(whoami) $TEGRA_CMD" &> /dev/null || true
    ssh $PC "pkill -u \$(whoami) $PC_CMD" &> /dev/null || true
}

mkdir -p logs

#Argument parsing
while [ $# -gt 0 ] ; do
    arg=$1
    shift

    case $arg in
        --clean)
            CLEAN="clean"
            ;;
        --args)
            PC_ARGS=$1
            shift
            ;;
        --tegra)
            TEGRA=$1
            shift
            ;;
        --pc)
            PC=$1
            shift
            ;;
        --reverse) #For debugging only. Not required to work
            T="$TEGRA_CMD"
            TEGRA_CMD="$PC_CMD"
            PC_CMD="$T"
            T="$TEGRA_ARGS"
            TEGRA_ARGS="$PC_ARGS"
            PC_ARGS="$T"
            ;;
        --repeat)
            RUNS=$1
    esac
done

if [ -z "$PC" ]; then
    if [ "$TEGRA" == "tegra-1" ]; then
        PC="in5050-2016-10"
        TEGRA="tegra-1"
    elif [ "$TEGRA" == "tegra-2" ]; then
        PC="in5050-2016-10"
        TEGRA="tegra-2"
    elif [ "$TEGRA" == "tegra-3" ]; then
        PC="in5050-2016-10"
        TEGRA="tegra-3"
    elif [ "$TEGRA" == "tegra-4" ]; then
        PC="in5050-2016-10"
        TEGRA="tegra-4"
    else
        echo "unknown tegra $TEGRA"
        exit 1
    fi
fi

TEGRA_NODE=$(/opt/DIS/sbin/disinfo get-nodeid -hostname ${TEGRA})
PC_NODE=$(/opt/DIS/sbin/disinfo get-nodeid -hostname ${PC})

echo "Using $TEGRA and $PC"

echo "Syncing source"
rsync ${RSYNC_ARGS} ${SRC_DIR}/ $TEGRA:${BUILD_DIR}/
rsync ${RSYNC_ARGS} ${SRC_DIR}/ $PC:${BUILD_DIR}/

#Compile on tegra and pc
echo
echo "### Compiling on Tegra ###"
ssh -t $TEGRA "cd $BUILD_DIR/tegra-build && make ${CLEAN} $TEGRA_CMD" || exit $?
echo

echo "### Compiling on PC ###"
ssh -t $PC "cd $BUILD_DIR/x86-build && make ${CLEAN} $PC_CMD" || exit $?
echo

#Launch on both nodes
echo "### Running ###"

if [ $RUNS -eq 1 ]; then
    stdbuf -oL -eL ssh $TEGRA "cd $BUILD_DIR/tegra-build && time stdbuf -oL -eL ./$TEGRA_CMD -r $PC_NODE $TEGRA_ARGS; echo Tegra exit code: \$?" |& tee logs/$DATE-tegra.log &
    stdbuf -oL -eL ssh $PC "cd $BUILD_DIR/x86-build && time stdbuf -oL -eL ./$PC_CMD -r $TEGRA_NODE $PC_ARGS; echo PC exit code: \$?" |& tee logs/$DATE-pc.log &
    wait
else
    for i in $(seq $RUNS)
    do
        START=`date +%s.%N`
        ssh $TEGRA "cd $BUILD_DIR/tegra-build && ./$TEGRA_CMD -r $PC_NODE $TEGRA_ARGS" &> /dev/null &
        ssh $PC "cd $BUILD_DIR/x86-build && ./$PC_CMD -r $TEGRA_NODE $PC_ARGS" &> /dev/null &

        wait
        END=`date +%s.%N`

        RUNTIME=$( echo "$END - $START" | bc -l )
        echo $i/$RUNS: $RUNTIME s
        echo $RUNTIME >> "../profiling/temp.txt"
    done
    python3 ../profiling/process.py ../profiling/temp.txt
    rm ../profiling/temp.txt
fi

quit
