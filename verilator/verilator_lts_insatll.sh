# uncomment if specific version is needed
# VERILATOR_VERSION=v5.020
PREFIX=$HOME/.usr

pushd /tmp/
git clone https://github.com/verilator/verilator.git
cd verilator

# uncomment if specific version is needed
# git checkout $VERILATOR_VERSION

git pull         # Make sure git repository is up-to-date
git tag          # See what versions exist

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
