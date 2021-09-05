#!/bin/bash 
while getopts n:u:x:y:h:m flag
do
    case "${flag}" in
        n) agentname=${OPTARG};;
        u) use_yaml_spawn=${OPTARG};;
        x) init_x=${OPTARG};;
        y) init_y=${OPTARG};;
	h) heading=${OPTARG};;
	m) map_name=${OPTARG};;
    esac
done

roslaunch planner.launch agentname:=$agentname use_yaml_spawn:=$use_yaml_spawn init_x:=$init_x init_y:=$init_y heading:=$heading map_name:=$map_name


