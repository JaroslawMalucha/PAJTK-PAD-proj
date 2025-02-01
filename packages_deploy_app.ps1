exit 1

##------------------------------------------------------------------------------
## redeploy from previously generated requirements file
##------------------------------------------------------------------------------

rm -r ./.venv
& "$Env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
./.venv/Scripts/Activate.ps1

python -m pip install --require-virtualenv --upgrade pip setuptools wheel IPython
pip install --require-virtualenv --requirement "./requirements-app.txt"




##------------------------------------------------------------------------------
## manual reinstall and package refreeze
##------------------------------------------------------------------------------

rm -r ./.venv
& "$Env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
./.venv/Scripts/Activate.ps1

pip list

python -m pip install --require-virtualenv --upgrade pip setuptools wheel IPython

pip list

pip install --require-virtualenv `
    streamlit `
    pandas `
    numpy `
    opendatasets `
    ipython `
    ipykernel `
    scikit-learn `
    matplotlib `
    seaborn `
    geopandas `
    contextily `
    geodatasets `
    imblearn `
    xgboost

pip list

pip freeze --require-virtualenv > "./requirements-app.txt"

