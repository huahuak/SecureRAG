install_sgx_lib() {
    if [ ! -d "lib" ]; then
        mkdir "lib"
    fi

    cp -u build/sgx/enclave.signed.so lib
    cp -u build/sgx/libenclave.so lib
}

. script/build.sh &&
    make -C build install &&
    install_sgx_lib
