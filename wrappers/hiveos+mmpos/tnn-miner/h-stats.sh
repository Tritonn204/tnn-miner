#!/usr/bin/env bash
source /hive/miners/custom/tnn-miner/h-manifest.sh


start_time=$(date +%s)

get_cpu_temps () {
  local t_core=`cpu-temp`
  local i=0
  local l_num_cores=$1
  local l_temp=
  for (( i=0; i < ${l_num_cores}; i++ )); do
    l_temp+="$t_core "
  done
  echo ${l_temp[@]} | tr " " "\n" | jq -cs '.'
}

get_cpu_fans () {
  local t_fan=0
  local i=0
  local l_num_cores=$1
  local l_fan=
  for (( i=0; i < ${l_num_cores}; i++ )); do
    l_fan+="$t_fan "
  done
  echo ${l_fan[@]} | tr " " "\n" | jq -cs '.'
}



get_log_time_diff(){
  local a=0
  let a=`date +%s`-`stat --format='%Y' $log_name`
  echo $a
}



uptime=$(get_miner_uptime)
[[ $uptime -lt 60 ]] && head -n 50 $log_name > $log_head_name
echo "miner uptime is: $uptime"

cpu_temp=`cpu-temp`
[[ $cpu_temp = "" ]] && cpu_temp=null

DATA=$(curl -s http://localhost:8989/stats)


total_khs=$(jq '.hashrate' <<< "$DATA")
total_khs=$(echo "scale=2; $total_khs / 1000" | bc)
khs=$total_khs
hs[0]=$total_khs
ac=$(jq '.accepted' <<< "$DATA")
rj=$(jq '.rejected' <<< "$DATA")
uptime=$(jq '.uptime' <<< "$DATA")
ver=$(jq '.version' <<< "$DATA")
echo "$ver"
algo="SPECTREX"
cpu_fan[0]=get_cpu_fans
hs_units="hs"
echo "HS[0] = ${hs[0]}"
echo "cputemp=${cpu_temp[0]}"


stats=$(jq -nc \
        --argjson total_khs "$total_khs" \
        --argjson khs "$total_khs" \
        --arg hs_units "$hs_units" \
        --argjson hs "$(echo "${hs[@]}" | jq -Rcs 'split(" ")')" \
        --argjson temp "$(echo "${cpu_temp[@]}" | jq -Rcs 'split(" ")')" \
        --argjson fan "$(echo "${cpu_fan[@]}" | jq -Rcs 'split(" ")')" \
        --arg uptime "$uptime" \
        --argjson ver "$ver" \
        --argjson ac "$ac" --argjson rj "$rj" \
        --arg algo "$algo" \
        '{$total_khs, $khs, $hs_units, $hs, $temp, $fan, $uptime, $ver, ar: [$ac, $rj], $algo }')

# debug output

 echo khs:   $hs
 echo stats: $stats
 echo ----------
