#!/usr/bin/env python3
"""
Live Log Monitor for Video Transcriber
Watches app_debug.log in real-time and displays new entries with color coding
"""

import time
import os
import sys
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️  watchdog not installed. Install with: pip install watchdog")
    print("⚠️  Falling back to polling mode...\n")


class LogMonitor(FileSystemEventHandler):
    """File system event handler for log file monitoring"""
    
    def __init__(self, log_file='app_debug.log'):
        self.log_file = log_file
        self.last_position = 0
        self.last_size = 0
        
        # Initialize position if file exists
        if os.path.exists(log_file):
            self.last_size = os.path.getsize(log_file)
            self.last_position = self.last_size
    
    def on_modified(self, event):
        """Called when the log file is modified"""
        if event.src_path.endswith(self.log_file):
            self.display_new_content()
    
    def display_new_content(self):
        """Read and display new content from log file"""
        if not os.path.exists(self.log_file):
            return
        
        current_size = os.path.getsize(self.log_file)
        
        # Only read if file has grown
        if current_size > self.last_size:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_content = f.read()
                
                if new_content.strip():
                    # Apply color coding
                    colored_content = self.colorize_log(new_content)
                    print(colored_content, end='')
                
                self.last_position = f.tell()
            
            self.last_size = current_size
    
    @staticmethod
    def colorize_log(text):
        """Add color coding to log messages"""
        lines = text.split('\n')
        colored_lines = []
        
        for line in lines:
            # Color code based on log level and emojis
            if 'ERROR' in line or '❌' in line or '🔴' in line:
                # Red for errors
                colored_lines.append(f"\033[91m{line}\033[0m")
            elif 'WARNING' in line or '⚠️' in line:
                # Yellow for warnings
                colored_lines.append(f"\033[93m{line}\033[0m")
            elif 'INFO' in line and ('✅' in line or '🚀' in line or '🎉' in line):
                # Green for success
                colored_lines.append(f"\033[92m{line}\033[0m")
            elif '=' in line and len(line) > 50:
                # Cyan for separator lines
                colored_lines.append(f"\033[96m{line}\033[0m")
            elif '🔍' in line or '📊' in line or '🤖' in line:
                # Blue for info
                colored_lines.append(f"\033[94m{line}\033[0m")
            else:
                # Default color
                colored_lines.append(line)
        
        return '\n'.join(colored_lines)


def poll_mode_monitor(log_file='app_debug.log', interval=0.5):
    """Fallback polling mode when watchdog is not available"""
    monitor = LogMonitor(log_file)
    
    print(f"📊 Polling {log_file} every {interval}s")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            monitor.display_new_content()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def watch_mode_monitor(log_file='app_debug.log'):
    """Watch mode using watchdog for efficient monitoring"""
    print(f"🔍 Starting Live Monitor for: {log_file}")
    print("Press Ctrl+C to stop\n")
    
    # Display initial separator
    print("=" * 80)
    print("📊 MONITORING STARTED")
    print("=" * 80)
    print()
    
    monitor = LogMonitor(log_file)
    observer = Observer()
    observer.schedule(monitor, path='.', recursive=False)
    observer.start()
    
    try:
        # Also poll periodically to catch any missed events
        while True:
            time.sleep(1)
            monitor.display_new_content()
    except KeyboardInterrupt:
        observer.stop()
        print("\n\n👋 Monitoring stopped")
    
    observer.join()


def display_summary():
    """Display helpful information about the monitor"""
    print("\n" + "=" * 80)
    print("📋 LOG MONITOR GUIDE")
    print("=" * 80)
    print()
    print("🔍 What to look for:")
    print("  - ✅ Green = Success messages")
    print("  - ❌ Red = Errors and failures")
    print("  - ⚠️  Yellow = Warnings")
    print("  - 🤖 Blue = AI/API operations")
    print("  - 📊 Cyan = Section separators")
    print()
    print("🎯 Key sections to watch:")
    print("  1. SERVER STARTING UP - Shows configuration on startup")
    print("  2. NEW JOB REQUEST - Shows incoming requests")
    print("  3. STARTING SUMMARY GENERATION - AI summary process")
    print("  4. CALLING ANTHROPIC API - The actual API call")
    print("  5. API CALL FAILED/SUCCESSFUL - Results")
    print()
    print("💡 Tips:")
    print("  - Look for the '❌ API CALL FAILED!' section for errors")
    print("  - Check 'Error type' and 'Error message' for details")
    print("  - Review 'ENVIRONMENT VERIFICATION' for config issues")
    print()
    print("=" * 80)
    print()


def main():
    """Main entry point"""
    log_file = 'app_debug.log'
    
    # Show initial guide
    display_summary()
    
    # Check if log file exists, if not create it
    if not os.path.exists(log_file):
        print(f"⚠️  Log file '{log_file}' not found. Creating it...")
        Path(log_file).touch()
        print(f"✅ Created {log_file}")
        print()
    
    # Show last 20 lines if file has content
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        print("📋 Last 20 lines from log:")
        print("=" * 80)
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                monitor = LogMonitor()
                for line in lines[-20:]:
                    print(monitor.colorize_log(line.rstrip()))
        except Exception as e:
            print(f"Could not read existing log: {e}")
        print("=" * 80)
        print()
    
    # Start monitoring
    if WATCHDOG_AVAILABLE:
        watch_mode_monitor(log_file)
    else:
        poll_mode_monitor(log_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting monitor")
        sys.exit(0)
