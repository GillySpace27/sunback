git commit -am %2
git tag -a "%1" -m "%1 : ""%2""
del dist\*
python setup.py sdist bdist_wheel
for /d /r . %%d in (Sunback-*) do @if exist "%%d" rd /s/q "%%d"