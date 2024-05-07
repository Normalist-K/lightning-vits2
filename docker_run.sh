name="--name $1"
vols="-v /home/hans/code/lightning-vits2:/vits2/"
gpu="--gpus all"
ipc="--ipc=host"
image="hans/vits2_lightning:1.0"
cmd="/bin/bash"

docker run -it $name $vols $gpu $ipc $image $cmd
