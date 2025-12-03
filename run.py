#!/usr/bin/env python3
"""
Run script for Scope - Token Probability Viewer
Starts the Flask backend and Next.js frontend servers.
"""

import subprocess
import sys
import os
import signal
import argparse
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_colored(msg, color=Colors.END):
    print(f"{color}{msg}{Colors.END}")

def get_project_root():
    return Path(__file__).parent.resolve()

def run_backend(project_root):
    """Start the Flask backend server."""
    print_colored("\n🚀 Starting Flask backend on http://localhost:5001", Colors.CYAN)
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development'
    return subprocess.Popen(
        [sys.executable, 'app.py'],
        cwd=project_root,
        env=env
    )

def run_frontend(project_root):
    """Start the Next.js frontend server."""
    frontend_dir = project_root / 'scope-frontend'
    
    if not frontend_dir.exists():
        print_colored(f"❌ Frontend directory not found: {frontend_dir}", Colors.RED)
        return None
    
    print_colored("\n🚀 Starting Next.js frontend on http://localhost:3000", Colors.CYAN)
    
    # Use npm on Windows, npm/npx on Unix
    if sys.platform == 'win32':
        npm_cmd = 'npm.cmd'
    else:
        npm_cmd = 'npm'
    
    return subprocess.Popen(
        [npm_cmd, 'run', 'dev'],
        cwd=frontend_dir,
        shell=(sys.platform == 'win32')
    )

def run_all(project_root):
    """Start both backend and frontend servers."""
    processes = []
    
    print_colored("=" * 50, Colors.HEADER)
    print_colored("       SCOPE - Token Probability Viewer", Colors.BOLD)
    print_colored("=" * 50, Colors.HEADER)
    
    # Start backend
    backend = run_backend(project_root)
    if backend:
        processes.append(('Backend', backend))
    
    # Start frontend
    frontend = run_frontend(project_root)
    if frontend:
        processes.append(('Frontend', frontend))
    
    if not processes:
        print_colored("❌ No servers started!", Colors.RED)
        return
    
    print_colored("\n" + "=" * 50, Colors.GREEN)
    print_colored("✅ Servers running!", Colors.GREEN)
    print_colored("   Backend:  http://localhost:5001", Colors.CYAN)
    print_colored("   Frontend: http://localhost:3000", Colors.CYAN)
    print_colored("=" * 50, Colors.GREEN)
    print_colored("\nPress Ctrl+C to stop all servers\n", Colors.YELLOW)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print_colored("\n\n🛑 Shutting down servers...", Colors.YELLOW)
        for name, proc in processes:
            if proc and proc.poll() is None:
                print_colored(f"   Stopping {name}...", Colors.CYAN)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print_colored("✅ All servers stopped.", Colors.GREEN)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes
    try:
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

def main():
    parser = argparse.ArgumentParser(
        description='Run Scope servers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py           # Start both backend and frontend
  python run.py --backend # Start only the Flask backend
  python run.py --frontend # Start only the Next.js frontend
        """
    )
    parser.add_argument('--backend', '-b', action='store_true', help='Run only the Flask backend')
    parser.add_argument('--frontend', '-f', action='store_true', help='Run only the Next.js frontend')
    
    args = parser.parse_args()
    project_root = get_project_root()
    
    # If neither flag is set, run both
    if not args.backend and not args.frontend:
        run_all(project_root)
    elif args.backend and args.frontend:
        run_all(project_root)
    elif args.backend:
        print_colored("=" * 50, Colors.HEADER)
        print_colored("       SCOPE - Backend Only", Colors.BOLD)
        print_colored("=" * 50, Colors.HEADER)
        proc = run_backend(project_root)
        if proc:
            try:
                proc.wait()
            except KeyboardInterrupt:
                print_colored("\n🛑 Stopping backend...", Colors.YELLOW)
                proc.terminate()
    elif args.frontend:
        print_colored("=" * 50, Colors.HEADER)
        print_colored("       SCOPE - Frontend Only", Colors.BOLD)
        print_colored("=" * 50, Colors.HEADER)
        proc = run_frontend(project_root)
        if proc:
            try:
                proc.wait()
            except KeyboardInterrupt:
                print_colored("\n🛑 Stopping frontend...", Colors.YELLOW)
                proc.terminate()

if __name__ == '__main__':
    main()

