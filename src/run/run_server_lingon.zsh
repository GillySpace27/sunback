#!/bin/zsh

echo $'\n>>>Bootstrapping Sunback<<<'

echo $'\n\t>Initing Conda'
source /Users/cgilbert/opt/anaconda3/etc/profile.d/conda.sh
source ~/.zshrc


# Run the remaining commands in a sub-shell
(
  echo $'\n\t>Changing Directory'
  cd /Users/cgilbert/PycharmProjects/sunback/src/
  echo $'\t'$PWD
  echo $'\n\t>Activating Sunback_env...'
  conda activate sunback_env

  echo $'\n\t>Prepending Paths...'
  export PATH=$PWD:$PATH
  export PYTHONPATH=$PWD:$PYTHONPATH

  echo $'\n\t>Running Server...\n'
  /Users/cgilbert/opt/anaconda3/envs/sunback_env/bin/python /Users/cgilbert/PycharmProjects/sunback/src/run/run_server_lingon.py

)
