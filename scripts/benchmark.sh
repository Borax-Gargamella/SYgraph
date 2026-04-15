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

benchs=("bfs" "sssp" "bc" "cc")
graphs=("hollywood-2009" "soc-orkut" "soc-LiveJournal1" "roadNet-CA" "kron_g500-logn21" "indochina-2004" "road_usa" "soc-twitter-2010")

declare -A SOURCES=(
  ["roadNet-CA"]="63595 69413 162712 215065 231882 242756 291628 313321 337137 380410 402093 416848 444743 499791 553050 611383 638912 758449 1008246 1319242"
  ["hollywood-2009"]="121253 1035649 602451 904248 865517 854930 31922 623076 807819 486820 320853 927132 517242 884131 247188 42013 214421 1002581 258499 59764"
  ["soc-orkut"]="698989 2688210 950649 659341 2612682 1854563 2629341 801571 2159475 545720 2873415 2219636 302210 657473 2851808 168592 2252915 2107278 301658 230095"
  ["soc-LiveJournal1"]="3872779 4826321 2059506 2266461 3494912 1107803 1912513 3779727 344173 446875 2593191 4575639 107249 2513291 2473871 4203909 1107036 4688748 751713 443344"
  ["kron_g500-logn21"]="608753 687052 989381 593082 403988 2062447 846121 624295 237579 639491 1685862 998599 1924764 756640 1763921 1118330 203738 773360 1845612 153582"
  ["indochina-2004"]="3984130 5904856 4075838 2512972 1751525 7057152 5988778 5845227 6558921 1684466 4697238 2528468 4209542 2590411 86450 6164426 1090065 579022 6415315 410319"
  ["road_usa"]="15461103 7942598 16420899 12528723 12571386 16170009 11736720 22964899 3939411 7320340 17422563 5891858 18925178 15123690 15923591 14754673 5814204 5288336 19117321 1961107"
  ["soc-twitter-2010"]="447625 1642633 974033 129222 1933298 344092 811158 232494 931048 223517 1190836 1051365 1694577 586027 1368976 1548584 952578 251119 306387 1740693"
)

function benchmark {
  bench=$1
  graph=$2
  graph_path=$datasets_path/$graph/$graph.bin

  sources=(${SOURCES[$graph]})

  mkdir -p $SCRIPT_DIR/logs/$bench

  echo Running $bench on $graph
  for source in "${sources[@]}"
  do
    echo -n '#'
    $SCRIPT_DIR/build/bin/$bench -b $graph_path -s $source -v >> $SCRIPT_DIR/logs/$bench/$graph.log 2>> $SCRIPT_DIR/logs/$bench/$graph.err
  done
  echo 
}

for bench in "${benchs[@]}"
do
  echo "Bench: $bench"
  for graph in "${graphs[@]}"
  do
    benchmark $bench $graph
  done
done
