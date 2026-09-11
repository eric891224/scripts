# extract a tar.gz archive to a destination directory, stripping the top-level directory from the archive
# Usage: extract_archive <label> <archive> <destination>
#   label: a human-readable label for the archive (used in error messages)
#   archive: the path to the tar.gz archive to extract
#   destination: the path to the directory where the archive should be extracted

extract_archive() {
  local label="$1"
  local archive="$2"
  local destination="$3"

  if [[ -e "$destination" ]]; then
    printf 'error: extraction destination already exists: %s\n' \
      "$destination" \
      >&2
    return 1
  fi

  mkdir -p "$destination"

  printf 'Extracting %s\n' "$label"

  if ! tar \
    --extract \
    --gzip \
    --file "$archive" \
    --directory "$destination" \
    --strip-components=1 \
    --no-same-owner \
    --no-same-permissions
  then
    printf 'error: failed to extract %s\n' "$label" >&2
    rm -rf -- "$destination"
    return 1
  fi
}