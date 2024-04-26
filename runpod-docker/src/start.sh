#!/bin/bash

pgrep ollama | xargs kill

ollama serve 2>&1 | tee ollama.server.log &
# Store the process ID (PID) of the background command

check_server_is_running() {
    # Replace "Listening" with the actual expected output
    if tail -n 10 ollama.server.log | grep -q "Listening"; then
        return 0 # Success
    else
        return 1 # Failure
    fi
}

# Wait for the process to print "Listening"
while ! check_server_is_running; do
    echo "Waiting..."
    sleep 1
done

python -u main.py custom
