#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) 
SCRIPT_DIR="$SCRIPT_DIR/.."

datasets_path="/data/datasets"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d | --datasets)
      datasets_path=$2
      shift
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -d, --datasets <path>  Path to the datasets directory"
      echo "  -h, --help             Display this help message"
      return 0 2>/dev/null
      exit 0
      ;;
    *)
    echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

# check if the datasets path exists
if [ ! -d $datasets_path ]; then
  echo "Datasets path does not exist: $datasets_path"
  return 1 2>/dev/null
  exit 1
fi

graphs=("hollywood-2009" "soc-orkut" "soc-LiveJournal1" "kron_g500-logn21" "soc-twitter-2010")
betas=(1 10 100 1000 10000 100000 1000000 10000000 100000000)
alphas=(1 5 10 15 20 25 30 35 40)

declare -A SOURCES=(
  ["hollywood-2009"]="320853 927132 517242 884131 247188 42013 214421 1002581 258499 59764"
  ["soc-orkut"]="2873415 2219636 302210 657473 2851808 168592 2252915 2107278 301658 230095"
  ["soc-LiveJournal1"]="2593191 4575639 107249 2513291 2473871 4203909 1107036 4688748 751713 443344"
  ["kron_g500-logn21"]="687052 989381 593082 403988 2062447 846121 624295 237579 639491 1685862"
  ["soc-twitter-2010"]="1190836 1051365 1694577 586027 1368976 1548584 952578 251119 306387 1740693"
)

function benchmark {
  graph=$1
  alpha=$2
  beta=$3
  graph_path=$datasets_path/$graph/$graph.bin

  sources=(${SOURCES[$graph]})

  mkdir -p $SCRIPT_DIR/logs/bfs-hybrid

  log_file=$SCRIPT_DIR/logs/bfs-hybrid/${graph}.log
  err_file=$SCRIPT_DIR/logs/bfs-hybrid/${graph}.err

  echo Running $bench on $graph with alpha=$alpha beta=$beta
  for source in "${sources[@]}"
  do
    echo -n '#'
    $SCRIPT_DIR/build/bin/bfs \
      -b $graph_path \
      -s $source \
      -V \
      --advance hybrid \
      --alpha $alpha \
      --beta $beta \
      >> $log_file 2>> $err_file
  done
  echo 
}

for graph in "${graphs[@]}"
do
  for alpha in ${alphas[@]}
  do
    for beta in "${betas[@]}"
    do
      benchmark $graph $alpha $beta
    done
  done
done
