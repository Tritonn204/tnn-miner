#!/bin/bash

token=$1

if [[ "$token" == "" ]]; then
  echo "Must provide token"
  exit 1
fi

sudo apt update
sudo apt upgrade
curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" -o script.deb.sh
chmod +x ./script.deb.sh
sudo ./script.deb.sh
sudo apt install gitlab-runner

sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce
sudo usermod -aG docker gitlab-runner

sudo gitlab-runner register -n --url https://gitlab.com  --token ${token} --executor docker --description "US-West-2 docker" --docker-image "docker:27" --docker-privileged   --docker-volumes "/certs/client"
sudo systemctl status gitlab-runner.service
