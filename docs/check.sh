#!/bin/sh
# Verify a built document against the requirements that have caught mistakes
# before: a clean log, the body inside its page limit, and a bibliography whose
# rendered entries are exactly the keys the text cites.
#
# Usage: ./check.sh [stem ...]     default: every stem with a built PDF
#
# Exits non-zero if any check fails. Set NO_COLOR=1 for plain output.

set -u
LC_ALL=C
export LC_ALL

cd "$(dirname "$0")" || exit 1

if [ -n "${NO_COLOR:-}" ] || [ "${TERM:-}" = dumb ]; then
    BOLD='' GREEN='' RED='' AMBER='' DIM='' RESET=''
else
    BOLD='\033[1m' GREEN='\033[32m' RED='\033[31m' AMBER='\033[33m'
    DIM='\033[2m' RESET='\033[0m'
fi

pass() { printf "   ${GREEN}pass${RESET}  %s\n" "$1"; }
fail() { printf "   ${RED}FAIL${RESET}  %s\n" "$1"; }
note() { printf "   ${DIM}%s${RESET}\n" "$1"; }
pages() { if [ "$1" -eq 1 ]; then echo page; else echo pages; fi; }

# Body page limits, excluding references. Interim is a fixed 2; the rest are
# ceilings.
limit_for() {
    case "$1" in
        interim)  echo 2 ;;
        proposal) echo 4 ;;
        mid)      echo 8 ;;
        final)    echo 8 ;;
        *)        echo 0 ;;
    esac
}

stems="${*:-}"
if [ -z "$stems" ]; then
    for f in interim proposal mid final; do
        [ -f "$f.pdf" ] && stems="$stems $f"
    done
fi
if [ -z "$stems" ]; then
    printf "${RED}no built PDFs; run make -C docs all from the repository root first${RESET}\n" >&2
    exit 1
fi

failed=0

for stem in $stems; do
    printf "${BOLD}%s${RESET}\n" "$stem"
    if [ ! -f "$stem.pdf" ]; then
        fail "no $stem.pdf"
        failed=1
        continue
    fi

    log="$stem.log"
    errors=$(grep -ac '^! ' "$log")
    overfull=$(grep -ac 'Overfull' "$log")
    warnings=$(grep -ac 'LaTeX Warning' "$log")
    undefined=$(grep -ac 'undefined' "$log")
    summary="errors $errors, overfull $overfull, warnings $warnings, undefined $undefined"

    if [ "$errors" -eq 0 ] && [ "$overfull" -eq 0 ] && [ "$warnings" -eq 0 ] \
        && [ "$undefined" -eq 0 ]; then
        pass "log clean"
    else
        fail "log: $summary"
        failed=1
    fi

    # Body length counts every page holding body text. References starting at the
    # top of its page means the body ended on the page before; References starting
    # part way down means the body spilled onto that page and it counts too. Reading
    # only where References appears would pass a body that overflows by half a page.
    total=$(pdftotext "$stem.pdf" - 2>/dev/null | awk -v RS='\f' 'END{print NR}')
    refpage=$(pdftotext "$stem.pdf" - 2>/dev/null \
        | awk -v RS='\f' '/References/{print NR; exit}')
    spill=0
    if [ -n "$refpage" ]; then
        first=$(pdftotext "$stem.pdf" - 2>/dev/null \
            | awk -v RS='\f' -v p="$refpage" 'NR==p' | awk 'NF{print; exit}')
        if [ "$first" = "References" ]; then
            body=$((refpage - 1))
        else
            body="$refpage"
            spill=$(pdftotext "$stem.pdf" - 2>/dev/null | awk -v RS='\f' -v p="$refpage" 'NR==p' \
                | awk '/^References$/{exit} NF{n++} END{print n+0}')
        fi
    else
        body="$total"
    fi
    limit=$(limit_for "$stem")

    if [ "$stem" = interim ] && [ "$body" -ne 2 ]; then
        fail "body is $body $(pages "$body"); the interim body must be exactly 2"
        failed=1
    elif [ "$limit" -gt 0 ] && [ "$body" -gt "$limit" ]; then
        fail "body reaches page $body, over the limit of $limit, spilling $spill lines"
        failed=1
    else
        pass "body $body $(pages "$body"), limit $limit, $total in total"
    fi

    # Cited keys against rendered bibliography entries, both directions.
    if [ -f "$stem.bbl" ]; then
        grep -aohE '\\cite[a-z]*\{[^}]*\}' "$stem.tex" \
            | grep -aoE '\{.*\}' | tr -d '{}' | tr ',' '\n' | tr -d ' ' \
            | sort -u > ".check-cited"
        tr '\n' ' ' < "$stem.bbl" \
            | grep -aoE '\\bibitem\[[^]]*\]\{[^}]*\}' \
            | grep -aoE '\{[^}]*\}$' | tr -d '{}' | sort -u > ".check-rendered"
        cited=$(wc -l < ".check-cited" | tr -d ' ')
        rendered=$(wc -l < ".check-rendered" | tr -d ' ')
        if diff ".check-cited" ".check-rendered" > ".check-diff"; then
            if [ "$cited" -eq 0 ]; then
                note "no citations yet"
            else
                pass "bibliography, $cited keys cited and rendered"
            fi
        else
            fail "cited keys and rendered entries differ:"
            sed "s/^/         /" ".check-diff"
            failed=1
        fi
        rm -f ".check-cited" ".check-rendered" ".check-diff"
    else
        note "no .bbl yet"
    fi
done

if [ "$failed" -ne 0 ]; then
    printf "${RED}${BOLD}failed${RESET}\n"
    exit 1
fi
printf "${GREEN}all checks passed${RESET}\n"
