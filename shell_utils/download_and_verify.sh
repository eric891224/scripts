# download_and_verify
#        │
#        ▼
#      curl
#        │
#    ┌───┴────┐
#   失敗      成功
#    │         │
# 刪檔案       ▼
# return 1   sha256sum
#              │
#              ▼
#        和 expected 比較
#           │        │
#         不同       相同
#           │         │
#        刪檔案     Verified
#        return 1    return 0

download_and_verify() {
  local label="$1"
  local url="$2"
  local expected_sha256="$3"
  local destination="$4"

  printf 'Downloading %s\n' "$label"

  if ! curl \
    --fail \
    --location \
    --retry 3 \
    --silent \
    --show-error \
    --output "$destination" \
    "$url"
  then
    rm -f -- "$destination"
    printf 'error: failed to download %s\n' "$label" >&2
    return 1
  fi

  local checksum_output
  checksum_output="$(sha256sum "$destination")"

  local actual_sha256="${checksum_output%% *}"

  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'error: checksum mismatch for %s\n' "$label" >&2
    printf 'expected: %s\n' "$expected_sha256" >&2
    printf 'actual:   %s\n' "$actual_sha256" >&2
    rm -f -- "$destination"
    return 1
  fi

  printf 'Verified %s\n' "$label"
}