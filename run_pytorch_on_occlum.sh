#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
python_dir="$script_dir/occlum_instance/image/opt/python-occlum"

cd occlum_instance && rm -rf image
copy_bom -f ../pytorch.yaml --root image --include-dir /opt/occlum/etc/template

if [ ! -d $python_dir ];then
    echo "Error: cannot stat '$python_dir' directory"
    exit 1
fi

new_json="$(jq '.resource_limits.user_space_size = "60GB" |
                .resource_limits.kernel_space_heap_size = "1GB" |
                .resource_limits.max_num_of_threads = 1024 |
                .env.default += ["PYTHONHOME=/opt/python-occlum"]' Occlum.json)" && \
echo "${new_json}" > Occlum.json
occlum build

# Run the python demo
echo -e "${BLUE}occlum run /bin/python3 -m unittest test.test_rag_model.TESTFIDT5.generate${NC}"
occlum run /bin/python3 -m unittest test.test_rag_model.TestFIDT5.test_generate
