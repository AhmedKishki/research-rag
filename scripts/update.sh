#!/usr/bin/env bash
# Delegate to `research-rag update`, which is the one implementation of an update.
#
#   scripts/update.sh                 update: pull or upgrade, then report
#   scripts/update.sh /path/project   update, naming that project for the record
#   scripts/update.sh --check         report only; change nothing
#   scripts/update.sh --offline       do not touch the network (use with --check)
#
# The command behind this script stops every app the installation serves and
# prints the command that starts each one again; it does not start them.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
OFFLINE=0
PROJECT_ROOT=""

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^$/d'
}

for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --offline) OFFLINE=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) printf 'update.sh: unknown option: %s\n' "$arg" >&2; exit 2 ;;
    *) PROJECT_ROOT="$arg" ;;
  esac
done

COMMAND="$REPO_ROOT/.venv/bin/research-rag"
if [ ! -x "$COMMAND" ]; then
  COMMAND="$(command -v research-rag || true)"
fi
if [ -z "$COMMAND" ]; then
  printf 'update.sh: no research-rag command found. Run "uv sync" in %s, or\n' \
    "$REPO_ROOT" >&2
  printf 'update.sh: install the command with "uv tool install git+https://github.com/AhmedKishki/research-rag.git"\n' >&2
  exit 1
fi

# argparse reads the global options in front of the subcommand and that
# subcommand's own options behind it, so the two groups stay apart.
GLOBAL=()
LOCAL=()
[ -n "$PROJECT_ROOT" ] && GLOBAL+=(--project-root "$PROJECT_ROOT")
[ "$OFFLINE" = 1 ] && GLOBAL+=(--offline)
[ "$CHECK" = 1 ] || LOCAL+=(--apply)

exec "$COMMAND" ${GLOBAL[@]+"${GLOBAL[@]}"} update ${LOCAL[@]+"${LOCAL[@]}"}
