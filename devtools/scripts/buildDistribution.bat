git commit -am "%2"
git tag -a "%1" -m "%1 : %2"
git push
MOVE dist\* dist_old\
python setup.py sdist bdist_wheel
