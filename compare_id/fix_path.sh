relative_path=".."
full_path=$(readlink -f "$relative_path")

#export PYTHONPATH="${PYTHONPATH}:${full_path}"
export PYTHONPATH="${full_path}"
echo "PYTHONPATH: $PYTHONPATH"
