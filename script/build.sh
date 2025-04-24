do_cmake() {
    cmake -B build .
}

install_core() {
install_core_include() {
    if [ ! -d "include/core" ]; then
        mkdir -p "include/core"
    fi
    
    cp -u core/include/* include/core/
}
    make -C build/core install &&
    install_core_include
}

build() {
    install_core &&
        make -C build
}

do_cmake &&
    build
