#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"

print_server_instructions() {
  cat <<'EOF'
[server] Setup steps:
  cd MobiEdit/server/mobiedit
  conda create -n mobiedit-server python=3.9.7 -y
  conda activate mobiedit-server
  pip install -r requirements.txt
EOF
}

print_edge_instructions() {
  cat <<'EOF'
[edge] Build steps:
  cd MobiEdit/edge
  cmake -S . -B build
  cmake --build build -j
EOF
}

if [[ ! -d "${ROOT_DIR}/server" || ! -d "${ROOT_DIR}/edge" ]]; then
  echo "Expected directories not found under ${ROOT_DIR}"
  exit 1
fi

case "${TARGET}" in
  server)
    print_server_instructions
    ;;
  edge)
    print_edge_instructions
    ;;
  all)
    print_server_instructions
    echo
    print_edge_instructions
    ;;
  *)
    echo "Usage: $0 [server|edge|all]"
    exit 1
    ;;
esac
