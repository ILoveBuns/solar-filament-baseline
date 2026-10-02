#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${root_dir}/build/report}"
tectonic_bin="${TECTONIC:-tectonic}"

command -v "${tectonic_bin}" >/dev/null
command -v pdfinfo >/dev/null
command -v pdffonts >/dev/null
mkdir -p "${output_dir}"
cd "${root_dir}"

"${tectonic_bin}" -X compile report/main.tex \
  --outdir "${output_dir}" --keep-logs --keep-intermediates

pdf="${output_dir}/main.pdf"
log="${output_dir}/main.log"
pages="$(pdfinfo "${pdf}" | awk '/^Pages:/ {print $2}')"
if [[ "${pages}" != "4" ]]; then
  echo "report must be exactly 4 pages; got ${pages}" >&2
  exit 1
fi

if grep -E 'Overfull|undefined references|Citation .* undefined|Reference .* undefined' "${log}"; then
  echo "report contains layout or reference errors" >&2
  exit 1
fi

if pdffonts "${pdf}" | tail -n +3 | awk '$4 != "yes" {exit 1}'; then
  :
else
  echo "report contains a non-embedded font" >&2
  exit 1
fi

echo "verified: 4 pages, no overflow/undefined references, all fonts embedded"
