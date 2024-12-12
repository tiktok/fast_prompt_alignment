# Function to check and create a directory
create_directory() {
    local DIR=$1
    
    if [ ! -d "$DIR" ]; then
        mkdir -p "$DIR"
    fi
}

# Function to create a subdirectory in the base dir path
create_subdirectory() {
    local BASE_DIR=$1
    local SUBDIR=$2

    local FULL_PATH="${BASE_DIR}${SUBDIR}/"

    create_directory "$FULL_PATH"

    echo "$FULL_PATH"
}

