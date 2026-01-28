#!/bin/bash
# Test script for debug REST API endpoints
# Usage: npm run viz & sleep 5 && ./scripts/test-debug-api.sh

BASE_URL="http://localhost:5173"

echo "=== Debug API Test Script ==="
echo ""

echo "1. GET /__debug/state"
curl -s "$BASE_URL/__debug/state" | jq .
echo ""

echo "2. POST /__debug/step"
curl -s -X POST "$BASE_URL/__debug/step" | jq .
echo ""

echo "3. POST /__debug/reset"
curl -s -X POST "$BASE_URL/__debug/reset" | jq .
echo ""

echo "4. POST /__debug/config (switch to RWM)"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"algorithm":"rwm","stepSize":0.5}' \
  "$BASE_URL/__debug/config" | jq .
echo ""

echo "5. GET /__debug/state (verify RWM)"
curl -s "$BASE_URL/__debug/state" | jq .
echo ""

echo "6. POST /__debug/config (switch back to HMC, change distribution)"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"algorithm":"hmc","distribution":"banana"}' \
  "$BASE_URL/__debug/config" | jq .
echo ""

echo "7. GET /__debug/logs?limit=10"
curl -s "$BASE_URL/__debug/logs?limit=10" | jq .
echo ""

echo "=== Tests Complete ==="
