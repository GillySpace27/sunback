#!/bin/zsh

echo $'\n>>>Bootstrapping Sunback<<<'

echo $'\n\t>Initing zsh...'
source ~/.zshrc

# Run the remaining commands in a sub-shell
(
  echo $'\t>Changing Directory...'
  cd /Users/cgilbert/vscode/sunback/src/
  echo $'\t\t'$PWD

  echo $'\n\t>Activating Environment...'
  source ../.venv/bin/activate
  echo $'\t\t'$(which python)

  echo $'\n\t>Prepending CWD to Paths...'
  export PATH=$PWD:$PATH
  export PYTHONPATH=$PWD:$PYTHONPATH

  echo $'\n\t>Running Server file: "run_server_lingon.zsh"...\n'

  # Path to the timestamp file
  timestamp_file="/Users/cgilbert/vscode/sunback/src/run/run_server_lingon.timestamp"

  # Append the current date to the file
  date=$(date)
  echo -n "$date " >> "$timestamp_file"

  # save all the output from the following command to a log file, and also print to console
  /Users/cgilbert/vscode/sunback/.venv/bin/python /Users/cgilbert/vscode/sunback/src/run/run_server_lingon.py | tee /Users/cgilbert/vscode/sunback/src/run/run_server_lingon.log
  code_status=$?

  # Append success or failure flag based on command execution status
  if [[ $code_status -eq 0 ]]; then
      echo "SUCCESS" >> "$timestamp_file"
  else
      echo "FAIL" >> "$timestamp_file"
  fi

  echo Job Complete!
)
