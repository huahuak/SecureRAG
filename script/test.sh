export LD_LIBRARY_PATH=lib:/opt/intel/sgxsdk/lib64

python -m unittest test.test_ext.TestExt.test_copytensortosgx