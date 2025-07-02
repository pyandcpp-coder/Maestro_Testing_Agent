"""
Enhanced YAML Generator and Maestro Test Runner
This script connects to your mobile UI automation server, generates YAML from commands,
and runs Maestro tests with comprehensive error handling and reporting.
"""

import requests
import subprocess
import os
import json
import time
import re
from datetime import datetime
from pathlib import Path

# Configuration
POD_URL = ""  ## to be updated according to our server ip for 8000 port 
MAESTRO_TESTS_DIR = "maestro_tests"
REPORTS_DIR = "test_reports"
REQUEST_TIMEOUT = 60

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message"""
    print(f"{color}{message}{Colors.END}")

def print_header(title):
    """Print formatted header"""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"{title.center(60)}", Colors.BOLD + Colors.WHITE)
    print_colored(f"{'='*60}", Colors.CYAN)

def print_status(message, status="info"):
    """Print status message with appropriate color"""
    colors = {
        "success": Colors.GREEN + "✅ ",
        "error": Colors.RED + "❌ ",
        "warning": Colors.YELLOW + "⚠️ ",
        "info": Colors.BLUE + "ℹ️ ",
        "running": Colors.PURPLE + "🚀 "
    }
    print_colored(f"{colors.get(status, '')}{message}", Colors.WHITE)

def create_directories():
    """Create necessary directories"""
    Path(MAESTRO_TESTS_DIR).mkdir(exist_ok=True)
    Path(REPORTS_DIR).mkdir(exist_ok=True)
    print_status(f"Created directories: {MAESTRO_TESTS_DIR}, {REPORTS_DIR}", "success")

def check_server_health():
    """Check if the server is healthy and ready"""
    print_status("Checking server health...", "info")
    try:
        health_response = requests.get(f"{POD_URL}/health", timeout=10)
        health_response.raise_for_status()
        health_data = health_response.json()
        
        print_status(f"Server Status: {health_data.get('status', 'unknown')}", "success")
        print_status(f"Model Loaded: {health_data.get('model_loaded', False)}", "info")
        print_status(f"CUDA Available: {health_data.get('cuda_available', False)}", "info")
        print_status(f"Device: {health_data.get('device', 'unknown')}", "info")
        
        if not health_data.get('model_loaded', False):
            print_status("Warning: Model not fully loaded yet. This may take a few minutes.", "warning")
            return False
        return True
        
    except requests.exceptions.RequestException as e:
        print_status(f"Server health check failed: {e}", "error")
        return False

def sanitize_filename(command):
    """Create a safe filename from command"""
    # Remove special characters and limit length
    safe_name = re.sub(r'[^\w\s-]', '', command.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    timestamp = datetime.now().strftime("%H%M%S")
    return f"{safe_name[:25]}_{timestamp}"

def get_user_command():
    """Get command from user with validation"""
    print_header("YAML GENERATOR")
    
    while True:
        command = input(f"{Colors.CYAN}🎯 Enter command to generate YAML: {Colors.END}").strip()
        if command:
            return command
        print_status("Please enter a valid command.", "warning")

def generate_yaml(command):
    """Generate YAML from command using the API"""
    print_status(f"Sending request to: {POD_URL}/generate-yaml", "info")
    print_status(f"Command: '{command}'", "info")
    
    payload = {"command": command}
    
    try:
        response = requests.post(
            f"{POD_URL}/generate-yaml", 
            json=payload, 
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            print_status(f"Server returned error: {data['error']}", "error")
            return None
            
        yaml_content = data.get("yaml", "").strip()
        
        if not yaml_content:
            print_status("Received empty YAML content", "warning")
            return None
            
        return yaml_content
        
    except requests.exceptions.Timeout:
        print_status(f"Request timed out after {REQUEST_TIMEOUT} seconds", "error")
        return None
    except requests.exceptions.RequestException as e:
        print_status(f"Request failed: {e}", "error")
        return None
    except json.JSONDecodeError:
        print_status("Invalid JSON response from server", "error")
        return None

def save_yaml_file(yaml_content, command):
    """Save YAML content to file"""
    filename = sanitize_filename(command)
    file_path = Path(MAESTRO_TESTS_DIR) / f"{filename}.yml"
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        print_status(f"YAML saved to: {file_path}", "success")
        return file_path
        
    except IOError as e:
        print_status(f"Failed to save YAML file: {e}", "error")
        return None

def display_yaml_content(yaml_content):
    """Display the generated YAML content"""
    print_header("GENERATED YAML CONTENT")
    print_colored(yaml_content, Colors.GREEN)
    print_colored("-" * 60, Colors.CYAN)

def run_maestro_test(file_path):
    """Run Maestro test and capture results"""
    print_header("RUNNING MAESTRO TEST")
    print_status(f"Executing: maestro test {file_path}", "running")
    
    start_time = time.time()
    
    try:
        # Run maestro test and capture output
        result = subprocess.run(
            ["maestro", "test", str(file_path)], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Prepare test report data
        report_data = {
            "command": command,
            "file_path": str(file_path),
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        # Save detailed report
        save_test_report(report_data)
        
        # Display results
        if result.returncode == 0:
            print_status(f"Test completed successfully in {duration:.2f} seconds", "success")
        else:
            print_status(f"Test failed with return code {result.returncode}", "error")
        
        # Display output
        if result.stdout:
            print_header("TEST OUTPUT")
            print(result.stdout)
        
        if result.stderr:
            print_header("ERROR OUTPUT")
            print_colored(result.stderr, Colors.RED)
        
        return report_data
        
    except subprocess.TimeoutExpired:
        print_status("Test timed out after 5 minutes", "error")
        return None
    except FileNotFoundError:
        print_status("Maestro not found. Please ensure Maestro is installed and in PATH", "error")
        return None
    except Exception as e:
        print_status(f"Unexpected error running test: {e}", "error")
        return None

def save_test_report(report_data):
    """Save detailed test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(REPORTS_DIR) / f"test_report_{timestamp}.json"
    
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print_status(f"Test report saved: {report_file}", "info")
    except IOError as e:
        print_status(f"Failed to save report: {e}", "warning")

def send_analysis_request(report_data):
    """Send test results to analysis endpoint"""
    print_status("Sending results for analysis...", "info")
    
    analysis_payload = {
        "command": report_data["command"],
        "yaml": "",  # We could read the YAML file here if needed
        "stdout": report_data["stdout"],
        "stderr": report_data["stderr"],
        "success": report_data["success"]
    }
    
    try:
        response = requests.post(
            f"{POD_URL}/analyze",
            json=analysis_payload,
            timeout=30
        )
        response.raise_for_status()
        
        analysis = response.json()
        if "report" in analysis:
            print_header("AI ANALYSIS REPORT")
            print_colored(analysis["report"], Colors.GREEN)
        elif "error" in analysis:
            print_status(f"Analysis failed: {analysis['error']}", "warning")
            
    except requests.exceptions.RequestException as e:
        print_status(f"Analysis request failed: {e}", "warning")

def main():
    """Main execution function"""
    try:
        # Setup
        create_directories()
        
        # Check server health
        if not check_server_health():
            print_status("Server is not ready. Please try again later.", "error")
            response = input(f"{Colors.YELLOW}Do you want to continue anyway? (y/n): {Colors.END}").lower()
            if response not in ['y', 'yes']:
                return
        
        # Get user input
        global command
        command = get_user_command()
        # Generate YAML
        yaml_content = generate_yaml(command)
        if not yaml_content:
            print_status("Failed to generate YAML. Exiting.", "error")
            return
        
        # Display and save YAML
        display_yaml_content(yaml_content)
        file_path = save_yaml_file(yaml_content, command)
        if not file_path:
            return
        
        # Ask user if they want to run the test
        # print_colored(f"\n{Colors.CYAN}🤔 Do you want to run the Maestro test now? (y/n): {Colors.END}", end="")
        print_colored(f"\n{Colors.CYAN}🤔 Do you want to run the Maestro test now? (y/n): {Colors.END}")
        run_test = input().lower().strip()
        
        if run_test in ['y', 'yes']:
            # Run Maestro test
            report_data = run_maestro_test(file_path)
            
            if report_data:
                # Ask for AI analysis
                # print_colored(f"\n{Colors.CYAN}🤖 Do you want AI analysis of the test results? (y/n): {Colors.END}", end="")
                print_colored(f"\n{Colors.CYAN}🤖 Do you want AI analysis of the test results? (y/n): {Colors.END}", end="")

                analyze = input().lower().strip()
                
                if analyze in ['y', 'yes']:
                    send_analysis_request(report_data)
        
        print_header("PROCESS COMPLETED")
        print_status("All operations completed successfully!", "success")
        
    except KeyboardInterrupt:
        print_status("\nProcess interrupted by user", "warning")
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")

if __name__ == "__main__":
    main()