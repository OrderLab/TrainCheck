third_party_libs_dir=$(grep -oP '(?<=third_party_libs_dir=).+' example_pipelines/INSTALL_CONFIG)

# print the third_party_libs_dir
echo "third party libraries would be installed at: $third_party_libs_dir"

pkg_name="lightning-thunder"
venv_name="lightning-thunder"

buggy_commit="c816506"
# fixed_commit="bdb59d7"
fixed_pr="810"

# check directory

current_dir=$(pwd)

# check if the current dir contains mldaikon
if [ -d "mldaikon" ]; then
    echo "install.sh executed in the mldaikon directory"
else
    echo "install.sh executed in the wrong directory, please execute it in the mldaikon directory"
    exit 1
fi


install() {
    conda create --name ${venv_name} python==3.10
    conda activate ${venv_name}

    echo "virtual environment ${venv_name} created and activated"

    # get the current dir


    # install mldaikon to the new venv
    python -m pip install -e .

    ### Installing the dependencies for the pipeline

    # 1) Install nvFuser and PyTorch dependencies:
    python -m pip install --pre nvfuser-cu121-torch24

    # install Thunder from source
    git clone https://github.com/Lightning-AI/lightning-thunder.git
    cd ${pkg_name}
    python -m pip install -e .
}

build_buggy(){
    check_installation
    cd ${pkg_name}
    # if have defined the buggy version, checkout to the buggy version
    if [ -n "$buggy_commit" ]; then
        git checkout ${buggy_commit}
    elif [ -n "$buggy_pr" ]; then
        git fetch origin pull/${buggy_pr}/head:buggy_pr_branch
        git checkout buggy_pr_branch
    fi

    # add customization logic below
}

build_fixed(){
    check_installation
    cd ${pkg_name}
    # if have defined the fixed version, checkout to the fixed version
    if [ -n "$fixed_commit" ]; then
        git checkout ${fixed_commit}
    elif [ -n "$fixed_pr" ]; then
        git fetch origin pull/${fixed_pr}/head:fixed_pr_branch
        git checkout fixed_pr_branch
    fi

    # add customization logic below
}

check_installation() {
    # check if the virtual environment exists
    if conda env list | grep -q ${venv_name}; then
        echo "virtual environment ${venv_name} exists"
    else
        echo "virtual environment ${venv_name} does not exist, begin installation"
        install
    fi
    # check if the cloned repo exists
    if [ -d ${pkg_name} ]; then
        echo "repo ${pkg_name} exists"
    else
        echo "repo ${pkg_name} does not exist, begin installation"
        uninstall
        install
    fi
}

uninstall() {
    if conda env list | grep -q ${venv_name}; then
        conda deactivate
        conda remove --name ${venv_name} --all
        echo "virtual environment ${venv_name} removed"
    fi
    
    # remove the cloned repo if it exists
    if [ -d ${pkg_name} ]; then
        rm -rf ${pkg_name}
    fi
}

# check out the argument passed, could be "install", "uninstall", "build_buggy", and "build_fixed"

if [ "$1" = "install" ]; then
    echo "installing the dependencies for the pipeline"
    install
elif [ "$1" = "uninstall" ]; then
    echo "uninstalling the dependencies for the pipeline"
    uninstall
elif [ "$1" = "build_buggy" ]; then
    echo "building the buggy version of the pipeline"
    build_buggy
elif [ "$1" = "build_fixed" ]; then
    echo "building the fixed version of the pipeline"
    build_fixed
else
    echo "invalid argument, please use 'install', 'uninstall', 'build_buggy', or 'build_fixed'"
    exit 1
fi

cd ${current_dir}