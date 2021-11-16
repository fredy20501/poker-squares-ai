#!/usr/bin/env bash

javac *.java
nohup java TestRunner > /dev/null 2>&1
echo $! > save_pid.txt
