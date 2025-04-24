do_cmake() {
    cmake -B build .
}

install_core() {
    echo "***********************************************"
    echo "****             Install  Core             ****"
    echo "***********************************************"
    install_sgx_api() {
        if [ ! -d "include/sgx" ]; then
            mkdir "include/sgx"
        fi

        cp -u sgx/enclave/enclave.h include/sgx

    }
    install_core_include() {
        if [ ! -d "include/core" ]; then
            mkdir "include/core"
        fi

        cp -u core/include/* include/core
    }
    make -C build/core install &&
        install_core_include
}

install_sgx() {
    echo "***********************************************"
    echo "****              Install SGX              ****"
    echo "***********************************************"
    install_sgx_include() {
        if [ ! -d "include/sgx" ]; then
            mkdir "include/sgx"
        fi

        cp -u sgx/app/sgx_securerag.h include/sgx

    }
    make -C build/sgx &&
        install_sgx_include
}

install_ext() {
    echo "***********************************************"
    echo "****              Install Ext              ****"
    echo "***********************************************"

    make -C build/ext
}

build() {
    install_core &&
        install_sgx &&
        install_ext
}

do_cmake &&
    build
