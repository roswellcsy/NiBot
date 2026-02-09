#!/usr/bin/env bash
# NiBot E2E Test Suite -- automated API testing against a live deployment.
# Usage: bash scripts/e2e_test.sh [API_URL]
# Default: http://192.168.5.55:8080/api/chat

set -euo pipefail

# Bypass local proxy for LAN targets
export no_proxy="192.168.5.55,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

API="${1:-http://192.168.5.55:8080/api/chat}"
PASS=0
FAIL=0
SKIP=0
RESULTS=()
START_TIME=$(date +%s)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# ---- Helpers ----

run_test() {
    local id="$1" name="$2" prompt="$3" chat_id="$4" timeout="$5" check="$6"
    local check_mode="${7:-contains}"  # contains | not_contains | status | regex | has_content

    printf "  %-8s %-30s " "$id" "$name"

    local http_code body
    local tmpfile
    tmpfile=$(mktemp)

    if [ "$check_mode" = "raw_post" ]; then
        # Raw body mode -- prompt IS the raw body
        http_code=$(curl -s -o "$tmpfile" -w '%{http_code}' \
            -X POST "$API" \
            -H "Content-Type: application/json" \
            -d "$prompt" \
            --max-time "$timeout" 2>/dev/null) || http_code="000"
    else
        http_code=$(curl -s -o "$tmpfile" -w '%{http_code}' \
            -X POST "$API" \
            -H "Content-Type: application/json" \
            -d "{\"content\":$(echo "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'),\"sender_id\":\"e2e\",\"chat_id\":\"$chat_id\",\"timeout\":$timeout}" \
            --max-time "$((timeout + 5))" 2>/dev/null) || http_code="000"
    fi

    body=$(cat "$tmpfile" 2>/dev/null || echo "")
    rm -f "$tmpfile"

    local result="FAIL"
    case "$check_mode" in
        raw_post|contains)
            if echo "$body" | grep -qi "$check"; then
                result="PASS"
            fi
            ;;
        not_contains)
            if ! echo "$body" | grep -qi "$check"; then
                result="PASS"
            fi
            ;;
        status)
            if echo "$body" | grep -q "\"status\":.*$check\|\"error\""; then
                result="PASS"
            fi
            # For error status codes, check http_code too
            if [ "$http_code" = "$check" ]; then
                result="PASS"
            fi
            ;;
        has_content)
            if [ -n "$body" ] && echo "$body" | grep -q '"content"'; then
                result="PASS"
            fi
            ;;
        regex)
            if echo "$body" | grep -qE "$check"; then
                result="PASS"
            fi
            ;;
        cjk)
            if python3 -c "import sys; sys.exit(0 if any(0x4e00<=ord(c)<=0x9fff or 0x3040<=ord(c)<=0x30ff for c in sys.stdin.read()) else 1)" <<< "$body"; then
                result="PASS"
            fi
            ;;
    esac

    if [ "$result" = "PASS" ]; then
        printf "${GREEN}PASS${NC}\n"
        PASS=$((PASS + 1))
    else
        printf "${RED}FAIL${NC}  (http=$http_code body=${body:0:120})\n"
        FAIL=$((FAIL + 1))
    fi
    RESULTS+=("{\"id\":\"$id\",\"name\":\"$name\",\"result\":\"$result\",\"http\":\"$http_code\",\"body\":$(echo "${body:0:200}" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '""')}")
}

run_multi_turn() {
    local id="$1" name="$2" chat_id="$3" timeout="$4" check="$5"
    shift 5
    # Remaining args are prompt1 prompt2 ...
    local prompts=("$@")
    local body=""

    printf "  %-8s %-30s " "$id" "$name"

    for prompt in "${prompts[@]}"; do
        local tmpfile
        tmpfile=$(mktemp)
        curl -s -o "$tmpfile" \
            -X POST "$API" \
            -H "Content-Type: application/json" \
            -d "{\"content\":$(echo "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))'),\"sender_id\":\"e2e\",\"chat_id\":\"$chat_id\",\"timeout\":$timeout}" \
            --max-time "$((timeout + 5))" 2>/dev/null || true
        body=$(cat "$tmpfile" 2>/dev/null || echo "")
        rm -f "$tmpfile"
    done

    # Check LAST response
    if echo "$body" | grep -qi "$check"; then
        printf "${GREEN}PASS${NC}\n"
        PASS=$((PASS + 1))
    else
        printf "${RED}FAIL${NC}  (last_body=${body:0:120})\n"
        FAIL=$((FAIL + 1))
    fi
}

section() {
    echo ""
    echo "=== $1 ==="
}

# ---- Pre-flight ----

echo "NiBot E2E Test Suite"
echo "API: $API"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Health check
printf "  Health   %-30s " "API reachable"
HEALTH=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API" \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"ping\",\"sender_id\":\"e2e\",\"chat_id\":\"health_$(date +%s)\",\"timeout\":30}" \
    --max-time 35 2>/dev/null) || HEALTH="000"
if [ "$HEALTH" = "200" ]; then
    printf "${GREEN}PASS${NC}\n"
    PASS=$((PASS + 1))
else
    printf "${RED}FAIL${NC} (http=$HEALTH)\n"
    echo "API not reachable. Aborting."
    exit 1
fi

# ======== A: Basic Intelligence ========
section "A: Basic Intelligence (no tools)"

run_test "A-01" "Self introduction"     "Hello, who are you?"                              "e2e_a01" 30 "bot\|nibot\|assistant\|NiBot"
run_test "A-02" "Math calculation"      "What is 17 * 23? Answer with just the number."    "e2e_a02" 30 "391"
run_test "A-03" "Knowledge Q&A"         "Explain what a linked list is in 2 sentences."    "e2e_a03" 30 "node\|pointer\|link"
run_test "A-04" "Code generation"       "Write a Python function that reverses a string. Just show code." "e2e_a04" 30 "def"
run_test "A-05" "Multilingual"          "Translate 'Good morning' to Chinese and Japanese." "e2e_a05" 30 "" "cjk"

# ======== K: Error Handling (quick tests) ========
section "K: Error Handling (quick)"

run_test "K-01" "Empty content"         '{"content":"","sender_id":"e2e","chat_id":"k01","timeout":10}' "e2e_k01" 10 "error\|empty" "raw_post"
run_test "K-02" "Invalid JSON"          'not json at all'                                  "e2e_k02" 10 "error\|invalid" "raw_post"
run_test "K-04" "Unicode/emoji"         "Tell me about the rocket emoji in Chinese"        "e2e_k04" 30 "" "has_content"
# K-08: Oversized payload (>1MB) -- use temp file since arg too large for CLI
printf "  %-8s %-30s " "K-08" "Oversized payload"
K08_TMP=$(mktemp)
python3 -c "import json; print(json.dumps({'content':'x'*1100000,'sender_id':'e2e','chat_id':'k08','timeout':5}))" > "$K08_TMP"
K08_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API" -H "Content-Type: application/json" -d @"$K08_TMP" --max-time 10 2>/dev/null) || K08_CODE="000"
rm -f "$K08_TMP"
if [ "$K08_CODE" = "413" ] || [ "$K08_CODE" = "400" ]; then
    printf "${GREEN}PASS${NC}\n"
    PASS=$((PASS + 1))
else
    printf "${RED}FAIL${NC}  (http=$K08_CODE)\n"
    FAIL=$((FAIL + 1))
fi

# ======== B: File Operations ========
section "B: File Operations"

run_test "B-01" "List directory"        "List the files in my workspace directory."         "e2e_fileops" 60 "" "has_content"
run_test "B-02" "Write file"            "Create a file called test_hello.txt with the content 'Hello from E2E test'." "e2e_fileops" 60 "creat\|writ\|test_hello"
run_test "B-03" "Read file"             "Read the file test_hello.txt and show me its contents." "e2e_fileops" 60 "Hello from E2E"
run_test "B-04" "Edit file"             "In the file test_hello.txt, replace the word 'Hello' with 'Goodbye'." "e2e_fileops" 60 "replac\|edit\|updat\|Goodbye"
run_test "B-05" "Read edited file"      "Read test_hello.txt and show me its current contents." "e2e_fileops" 60 "Goodbye"
run_test "B-06" "Read nonexistent"      "Read the file /nonexistent/foo.txt"               "e2e_b06" 60 "not found\|error\|exist\|No such"
run_test "B-08" "Unicode write"         "Write a file test_unicode.txt with content: cafe中文テスト" "e2e_b08" 60 "writ\|creat"

# ======== C: Shell & System ========
section "C: Shell & System"

run_test "C-01" "Echo command"          "Run the command: echo Hello World"                "e2e_c01" 60 "Hello World"
run_test "C-02" "System info"           "What OS is this running on? Use the uname command." "e2e_c02" 60 "Linux"
run_test "C-03" "Python version"        "Check what Python version is installed."          "e2e_c03" 60 "3.13\|Python"
run_test "C-04" "Working directory"     "Run the pwd command and show the output."         "e2e_c04" 60 "workspace\|home\|nibot"
run_test "C-08" "Pipe command"          "Run: ls -la | head -5"                            "e2e_c08" 60 "total\|drwx\|rw"
run_test "C-09" "Disk space"            "Check disk space using df -h."                    "e2e_c09" 60 "Filesystem\|Size\|Use\|Avail"

# ======== D: Web & Information ========
section "D: Web & Information"

run_test "D-02" "Fetch public URL"      "Fetch the URL https://httpbin.org/get and show the response." "e2e_d02" 90 "headers\|url\|origin\|httpbin"
run_test "D-06" "DNS failure"           "Fetch https://thisdomaindoesnotexist12345.com/"   "e2e_d06" 60 "error\|fail\|resolve\|not found"

# ======== H: Admin Operations ========
section "H: Admin Operations"

run_test "H-01" "List config"           "Show the current NiBot configuration."            "e2e_h01" 30 "model\|temperature\|agent"
run_test "H-02" "Get model"             "What model are you currently using?"               "e2e_h02" 30 "claude\|sonnet\|anthropic"
run_test "H-05" "List schedules"        "List all scheduled tasks."                         "e2e_h05" 30 "schedule\|no\|job\|empty\|none"
run_test "H-08" "List skills"           "List all available skills."                        "e2e_h08" 30 "skill\|no\|empty\|none"

# ======== E: Code Quality ========
section "E: Code Quality"

run_test "E-02" "Git diff"              "Show the git diff for the workspace."             "e2e_e02" 60 "diff\|change\|no change\|clean\|no repo\|not.*git"

# ======== F: Scaffolding ========
section "F: Scaffolding"

run_test "F-01" "List templates"        "What project templates are available for scaffolding?" "e2e_f01" 30 "python-lib\|fastapi\|template"

# ======== I: Analytics ========
section "I: Analytics & Metrics"

run_test "I-01" "Session summary"       "Summarize recent sessions."                       "e2e_i01" 30 "session\|message\|recent\|no.*session"
run_test "I-05" "Search sessions"       "Search sessions for the word 'hello'."            "e2e_i05" 30 "search\|result\|found\|no\|hello"

# ======== J: Multi-turn Conversation ========
section "J: Multi-turn Conversation"

run_multi_turn "J-01" "Memory recall" "e2e_j01" 60 "Alice" \
    "My name is Alice. Please remember it." \
    "What is my name?"

run_multi_turn "J-02" "File context" "e2e_j02" 60 "Friday\|deadline" \
    "Write a file called notes.txt with content: deadline is Friday" \
    "What is in notes.txt?"

# Session isolation: different chat_ids should NOT share context
printf "  %-8s %-30s " "J-03" "Session isolation"
# Chat A: set a secret
curl -s -X POST "$API" -H "Content-Type: application/json" \
    -d '{"content":"The secret code is 42. Remember it.","sender_id":"e2e","chat_id":"e2e_j03a","timeout":30}' \
    --max-time 35 >/dev/null 2>&1
# Chat B: try to get it (different chat_id)
J03_BODY=$(curl -s -X POST "$API" -H "Content-Type: application/json" \
    -d '{"content":"What is the secret code?","sender_id":"e2e","chat_id":"e2e_j03b","timeout":30}' \
    --max-time 35 2>/dev/null)
if echo "$J03_BODY" | grep -q '"42"'; then
    printf "${RED}FAIL${NC}  (leaked: $J03_BODY)\n"
    FAIL=$((FAIL + 1))
else
    printf "${GREEN}PASS${NC}\n"
    PASS=$((PASS + 1))
fi

# ======== L: Cross-tool Complex Scenarios ========
section "L: Cross-tool Composite"

run_test "L-03" "System report"         "Check the Python version, disk space with df -h, and list installed pip packages. Save a summary to report.txt." "e2e_l03" 120 "report\|saved\|writ\|python\|pip"

# ======== K: Concurrency ========
section "K: Concurrency"

printf "  %-8s %-30s " "K-07" "Concurrent different sessions"
PIDS=()
TMPFILES=()
for i in $(seq 1 5); do
    tmp=$(mktemp)
    TMPFILES+=("$tmp")
    curl -s -o "$tmp" -X POST "$API" -H "Content-Type: application/json" \
        -d "{\"content\":\"What is $i + $i?\",\"sender_id\":\"e2e\",\"chat_id\":\"e2e_k07_$i\",\"timeout\":30}" \
        --max-time 35 &
    PIDS+=($!)
done
K07_PASS=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null || true
    if [ -s "${TMPFILES[$i]}" ]; then
        K07_PASS=$((K07_PASS + 1))
    fi
    rm -f "${TMPFILES[$i]}"
done
if [ "$K07_PASS" -ge 4 ]; then
    printf "${GREEN}PASS${NC} ($K07_PASS/5 succeeded)\n"
    PASS=$((PASS + 1))
else
    printf "${RED}FAIL${NC} ($K07_PASS/5 succeeded)\n"
    FAIL=$((FAIL + 1))
fi

# ======== Summary ========

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
TOTAL=$((PASS + FAIL + SKIP))

echo ""
echo "============================================"
echo "  E2E Test Results"
echo "============================================"
echo "  Total:    $TOTAL"
printf "  Passed:   ${GREEN}$PASS${NC}\n"
printf "  Failed:   ${RED}$FAIL${NC}\n"
if [ "$SKIP" -gt 0 ]; then
    printf "  Skipped:  ${YELLOW}$SKIP${NC}\n"
fi
echo "  Duration: ${DURATION}s"
echo "  Rate:     $(( PASS * 100 / (TOTAL > 0 ? TOTAL : 1) ))%"
echo "============================================"

# Write JSON results
RESULTS_JSON="["
for i in "${!RESULTS[@]}"; do
    if [ "$i" -gt 0 ]; then RESULTS_JSON+=","; fi
    RESULTS_JSON+="${RESULTS[$i]}"
done
RESULTS_JSON+="]"

echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"api\":\"$API\",\"total\":$TOTAL,\"passed\":$PASS,\"failed\":$FAIL,\"skipped\":$SKIP,\"duration_s\":$DURATION,\"results\":$RESULTS_JSON}" > e2e_results.json
echo "Results saved to e2e_results.json"

# Exit with failure if any test failed
[ "$FAIL" -eq 0 ]
