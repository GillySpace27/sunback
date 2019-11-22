git commit -am "%2"
git tag -a "%1" -m "%1 : %2"
MOVE dist\* dist_old\
python setup.py sdist bdist_wheel
::python -m twine upload dist/* -r testpypi
