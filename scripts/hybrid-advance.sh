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

graphs=("hollywood-2009" "soc-orkut" "soc-LiveJournal1" "roadNet-CA" "kron_g500-logn21" "indochina-2004" "road_usa" "soc-twitter-2010")
betas=(1 10 100 1000 10000 100000 1000000)
alphas=(1 5 10 15 20 25 30)

declare -A SOURCES=(
  ["roadNet-CA"]="63595"
  ["hollywood-2009"]="121253"
  ["soc-orkut"]="698989"
  ["soc-LiveJournal1"]="3872779"
  ["kron_g500-logn21"]="1448412" 
  ["indochina-2004"]="3984130"
  ["road_usa"]="15461103"
  ["soc-twitter-2010"]="447625"
)

function benchmark {
  graph=$1
  alpha=$2
  beta=$3
  graph_path=$datasets_path/$graph/$graph.bin

  sources=(${SOURCES[$graph]})

  mkdir -p $SCRIPT_DIR/logs/bfs-hybrid

  log_file=$SCRIPT_DIR/logs/bfs-hybrid/${graph}_alpha${alpha}_beta${beta}.log
  err_file=$SCRIPT_DIR/logs/bfs-hybrid/${graph}_alpha${alpha}_beta${beta}.err

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
