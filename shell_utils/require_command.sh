# EX.
# require_command docker
#         │
#         ├─ docker 有 → success
#         │
#         └─ docker 沒有 → error + return 127

require_command() {
  local command_name="$1"

  if command -v "$command_name" >/dev/null 2>&1; then
    return
  fi

  printf 'error: required command not found: %s\n' \
    "$command_name" \
    >&2
  return 127
}