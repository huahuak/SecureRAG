install_signed() {
    echo "***********************************************"
    echo "****            Install  Signed            ****"
    echo "***********************************************"
    if [ ! -d "lib" ]; then
        mkdir "lib"
    fi

    cp -u build/sgx/enclave.signed.so .
    # cp -u build/sgx/libenclave.so lib
}

. script/build.sh &&
    make -C build install &&
    install_signed
