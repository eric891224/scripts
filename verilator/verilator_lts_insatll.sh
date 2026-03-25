# ========== Required CLI Parameters ==========

usage() {
	echo "Usage: $0 --root_dir <path>"
	exit 1
}

ROOT_DIR=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		--root_dir)
			if [[ -z "$2" ]]; then
				echo "Error: --root_dir requires a value."
				usage
			fi
			ROOT_DIR="$2"
			shift 2
			;;
		-h|--help)
			usage
			;;
		*)
			echo "Error: Unknown argument '$1'."
			usage
			;;
	esac
done

if [[ -z "$ROOT_DIR" ]]; then
	echo "Error: required parameter --root_dir is missing."
	usage
fi

# ========== Installation for Verilator Stable Version ==========

# uncomment if specific version is needed
# VERILATOR_VERSION=v5.020
PREFIX="$ROOT_DIR"

pushd /tmp/
git clone -b stable --depth 1 https://github.com/verilator/verilator.git
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
rm -rf /tmp/verilator

# Set the PATH environment variable
export PATH=${PREFIX}/bin:$PATH
echo export PATH=${PREFIX}/bin:\$PATH >> ~/.bashrc
