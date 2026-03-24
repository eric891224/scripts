VERILATOR_VERSION=v5.020
PREFIX=$HOME/.usr

pushd /tmp/
git clone https://github.com/verilator/verilator.git
cd verilator
git checkout v5.020
# Every time you need to build:
unset VERILATOR_ROOT          # For bash
autoconf                      # Create ./configure script
./configure --prefix=$PREFIX  # Configure and create Makefile
make -j `nproc`
make install
popd
rm -r /tmp/verilator

# Set the PATH environment variable
export PATH=${PREFIX}/bin:$PATH
echo export PATH=${PREFIX}/bin:\$PATH >> ~/.bashrc
