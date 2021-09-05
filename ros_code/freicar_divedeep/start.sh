#!/bin/bash 
while getopts agent_name:use_yaml_spawn:x_spawn:y_spawn:heading:map_name flag
do
    case "${flag}" in
        agent_name) agent_name=${OPTARG};;
        use_yaml_spawn) use_yaml_spawn=${OPTARG};;
        x_spawn) init_x=${OPTARG};;
        y_spawn) init_y=${OPTARG};;
	heading) heading=${OPTARG};;
	map_name) map_name=${OPTARG};;
    esac
done

roslaunch freicar.launch agent_name:=freicar_2 use_yaml_spawn:=$use_yaml_spawn init_x:=$init_x init_y:=$init_y heading:=$heading map_name:=freicar_1.aismap


