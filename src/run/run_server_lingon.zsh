#!/bin/zsh

echo $'\n>>>Bootstrapping Sunback<<<'

echo $'\n\t>Initing zsh...'
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh
source ~/.zshrc

# Run the remaining commands in a sub-shell
(
  echo $'\t>Changing Directory...'
  cd /Users/cgilbert/vscode/sunback/src/
  echo $'\t\t'$PWD

  echo $'\n\t>Activating Sunback_env_arm...'
  conda activate sunback_env_arm
  echo $'\t\t'$(which python)

  echo $'\n\t>Prepending CWD to Paths...'
  export PATH=$PWD:$PATH
  export PYTHONPATH=$PWD:$PYTHONPATH

  echo $'\n\t>Running Server file: "run_server_lingon.zsh"...\n'
 /opt/homebrew/anaconda3/envs/sunback_env_arm/bin/python /Users/cgilbert/vscode/sunback/src/run/run_server_lingon.py

  echo Job Complete!
)
