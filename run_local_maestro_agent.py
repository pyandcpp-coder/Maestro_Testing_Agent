"""
Local Maestro Test Automation System
This script uses a local Ollama model to generate Maestro YAML from commands,
saves the flow, and executes it against a running application.
"""

import subprocess
import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
import ollama 

OLLAMA_MODEL_NAME = "maestro-agent"

MAESTRO_TESTS_DIR = "maestro_tests" 

REPORTS_DIR = "test_reports"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.CYAN):
    print(f"{color}{message}{Colors.END}")

def print_header(title):
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"{title.center(60)}", Colors.BOLD + Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

def print_status(message, status="info"):
    colors = {
        "success": Colors.GREEN + "✅ ",
        "error": Colors.RED + "❌ ",
        "warning": Colors.YELLOW + "⚠️ ",
        "info": Colors.BLUE + "ℹ️ ",
        "running": Colors.PURPLE + "🚀 "
    }
    print(f"{colors.get(status, '')}{message}")

def create_directories():
    """Create necessary directories if they don't exist."""
    Path(MAESTRO_TESTS_DIR).mkdir(exist_ok=True)
    Path(REPORTS_DIR).mkdir(exist_ok=True)
    print_status(f"Ensured directories exist: {MAESTRO_TESTS_DIR}, {REPORTS_DIR}", "success")

def check_ollama_running():
    """Check if the Ollama server is accessible."""
    try:
        ollama.ps() # This will raise an exception if Ollama is not running
        print_status("Ollama server is running.", "success")
        return True
    except Exception:
        print_status("Could not connect to Ollama.", "error")
        print_status("Please make sure the Ollama application is running on your Mac.", "warning")
        return False

def generate_yaml_local(command):
    """Generate YAML from a command using the local Ollama model."""
    print_status(f"Sending command to local model: '{OLLAMA_MODEL_NAME}'", "running")
    print_status(f"Command: '{command}'", "info")


    system_prompt = """You are an expert Maestro automation assistant. Given a user request in plain English, you must generate the corresponding YAML code to be used in a Maestro flow. Only output the YAML code inside a code block. Do not add any other explanations or text."""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': command},
            ],
            options={'temperature': 0.1} 
        )
        
        raw_content = response['message']['content']
        yaml_match = re.search(r"```(?:yaml)?\n(.*?)\n```", raw_content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
        else:
            yaml_content = raw_content.strip()

        if not yaml_content:
            print_status("Model returned an empty response.", "warning")
            return None
        
        return yaml_content

    except Exception as e:
        print_status(f"Failed to generate YAML with Ollama: {e}", "error")
        return None

def sanitize_filename(command):
    """Create a safe filename from the command."""
    safe_name = re.sub(r'[^\w\s-]', '', command.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name).strip('_')
    timestamp = datetime.now().strftime("%H%M%S")
    return f"flow_{safe_name[:30]}_{timestamp}"


def save_yaml_file(yaml_content, command):
    """Saves the YAML content directly as provided by the LLM."""
    filename = sanitize_filename(command)
    file_path = Path(MAESTRO_TESTS_DIR) / f"{filename}.yml"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print_status(f"YAML flow saved to: {file_path}", "success")
        return file_path
    except IOError as e:
        print_status(f"Failed to save YAML file: {e}", "error")
        return None

def run_maestro_test(file_path):
    """Run Maestro test and handle output."""
    print_header("RUNNING MAESTRO TEST")
    command_to_run = ["maestro", "test", str(file_path)]
    print_status(f"Executing: {' '.join(command_to_run)}", "running")
    
    try:
        process = subprocess.Popen(command_to_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout_lines, stderr_lines = [], []
        
        print_colored("-" * 60, Colors.PURPLE)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                stdout_lines.append(output)
        stderr_output = process.stderr.read()
        if stderr_output:
            print_colored("--- ERRORS ---", Colors.RED)
            print_colored(stderr_output.strip(), Colors.RED)
            stderr_lines.append(stderr_output)
        
        print_colored("-" * 60, Colors.PURPLE)
        
        if process.returncode == 0:
            print_status(f"Test completed successfully!", "success")
        else:
            print_status(f"Test failed with return code {process.returncode}", "error")
            
    except FileNotFoundError:
        print_status("Maestro not found. Is it installed and in your system's PATH?", "error")
    except Exception as e:
        print_status(f"An unexpected error occurred while running the test: {e}", "error")

def main():
    """Main execution loop."""
    try:
        print_header("LOCAL MAESTRO AGENT")
        create_directories()
        
        if not check_ollama_running():
            return
        command = input(f"{Colors.YELLOW}🎯 Enter command to generate Maestro flow: {Colors.END}").strip()
        if not command:
            print_status("No command entered. Exiting.", "warning")
            return
            
        yaml_content = generate_yaml_local(command)
        if not yaml_content:
            print_status("Failed to generate YAML. Aborting.", "error")
            return
            
        print_header("GENERATED YAML")
        print_colored(yaml_content, Colors.GREEN)
        file_path = save_yaml_file(yaml_content, command)
        if not file_path:
            return
        run_test = input(f"\n{Colors.YELLOW}🤔 Run this Maestro test now? (y/n): {Colors.END}").lower().strip()
        
        if run_test in ['y', 'yes']:
            run_maestro_test(file_path)
        else:
            print_status("Test execution skipped.", "info")

        print_header("PROCESS COMPLETE")

    except KeyboardInterrupt:
        print_status("\nProcess interrupted by user.", "warning")
    except Exception as e:
        print_status(f"A critical error occurred: {e}", "error")

if __name__ == "__main__":
    main()