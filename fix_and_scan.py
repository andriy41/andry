import os
import subprocess


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stdout)
        print(result.stderr)
    else:
        print(result.stdout)


# Define the commands to run
commands = [
    "flake8 .",  # Lint the code
    "black .",  # Format the code
    "mypy .",  # Type check the code
]

# Run each command
for command in commands:
    run_command(command)
