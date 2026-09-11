# 有設定 CONTAINER_RUNTIME 嗎？
#             │
#        ┌────┴────┐
#       yes        no
#        │          │
# 是 podman/docker? │
#        │          │
#    ┌───┴───┐      │
#   yes      no     │
#    │        │      │
# 檢查有無   error  │
#    │               │
# 輸出指定值         │
#                    ▼
#              有 podman 嗎？
#               │        │
#              yes       no
#               │         │
#            用 podman    ▼
#                     有 docker 嗎？
#                      │       │
#                     yes      no
#                      │        │
#                   用 docker  error

detect_container_runtime() {
  if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    case "$CONTAINER_RUNTIME" in
      podman | docker)
        require_command "$CONTAINER_RUNTIME"
        printf '%s\n' "$CONTAINER_RUNTIME"
        return
        ;;
      *)
        printf 'error: unsupported CONTAINER_RUNTIME: %s\n' \
          "$CONTAINER_RUNTIME" \
          >&2
        return 2
        ;;
    esac
  fi

  if command -v podman >/dev/null 2>&1; then
    printf '%s\n' podman
    return
  fi

  if command -v docker >/dev/null 2>&1; then
    printf '%s\n' docker
    return
  fi

  printf '%s\n' \
    'error: podman or docker is required to build slang-hier' \
    >&2
  return 127
}