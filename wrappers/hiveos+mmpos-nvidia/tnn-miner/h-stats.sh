#!/usr/bin/env bash
source /hive/miners/custom/tnn-miner-cuda/h-manifest.sh


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

DATA=$(curl -s http://localhost:8991/stats)

# Per-GPU hashrates (H/s from API)
gpu_count=$(jq '.gpus | length // 0' <<< "$DATA")
hs=()
temp=()
fan=()
busid=()

if [[ "$gpu_count" -gt 0 ]]; then
  for (( i=0; i < gpu_count; i++ )); do
    gpu_hr=$(jq ".gpus[$i].hashrate // 0" <<< "$DATA")
    gpu_hr_khs=$(echo "scale=2; $gpu_hr / 1000" | bc)
    hs+=($gpu_hr_khs)
    # GPU temps/fans via gpu-stats if available, else use cpu-temp as fallback
    temp+=($cpu_temp)
    fan+=(0)
    gpu_busid=$(jq -r ".gpus[$i].pcie_id // \"gpu$i\"" <<< "$DATA")
    busid+=("$gpu_busid")
  done
else
  # Fallback: single aggregate hashrate
  total_hr=$(jq '.hashrate // 0' <<< "$DATA")
  total_khs=$(echo "scale=2; $total_hr / 1000" | bc)
  hs+=($total_khs)
  temp+=($cpu_temp)
  fan+=(0)
fi

total_khs=0
for h in "${hs[@]}"; do
  total_khs=$(echo "scale=2; $total_khs + $h" | bc)
done
khs=$total_khs

ac=$(jq '.accepted // 0' <<< "$DATA")
rj=$(jq '.rejected // 0' <<< "$DATA")
uptime=$(jq '.uptime // 0' <<< "$DATA")
ver=$(jq '.version // "unknown"' <<< "$DATA")
echo "$ver"
algo=$(jq -r '.algo // "UNKNOWN"' <<< "$DATA")
hs_units="khs"

echo "total_khs: $total_khs"
echo "gpu_count: $gpu_count"

stats=$(jq -nc \
        --argjson total_khs "$total_khs" \
        --argjson khs "$total_khs" \
        --arg hs_units "$hs_units" \
        --argjson hs "$(printf '%s\n' "${hs[@]}" | jq -cs '.')" \
        --argjson temp "$(printf '%s\n' "${temp[@]}" | jq -cs '.')" \
        --argjson fan "$(printf '%s\n' "${fan[@]}" | jq -cs '.')" \
        --arg uptime "$uptime" \
        --argjson ver "$ver" \
        --argjson ac "$ac" --argjson rj "$rj" \
        --arg algo "$algo" \
        '{$total_khs, $khs, $hs_units, $hs, $temp, $fan, $uptime, $ver, ar: [$ac, $rj], $algo }')

# debug output

 echo khs:   ${hs[@]}
 echo stats: $stats
 echo ----------
