"""
MPU6050 Serial Data Logger to CSV
Captures data from ESP32 serial monitor and saves to CSV file
Commands: 's' to start, 'x' to stop collection
"""

import serial
import csv
import time
from datetime import datetime

# Configuration
SERIAL_PORT = 'COM6'  # Change if needed
BAUD_RATE = 115200
OUTPUT_FILE = f'mpu_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def main():
    print("=== MPU6050 Data Logger ===")
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        print("Connected!")
        print("\nCommands:")
        print("  Press 's' and Enter - Send START command to ESP32")
        print("  Press 'x' and Enter - Send STOP command to ESP32")
        print("  Press 'q' and Enter - Quit this program")
        print(f"\nData will be saved to: {OUTPUT_FILE}\n")
        
        csv_file = None
        csv_writer = None
        collecting = False
        header_written = False
        
        while True:
            # Check for user input (non-blocking)
            import sys
            import select
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    cmd = msvcrt.getch().decode('utf-8').lower()
                    if cmd == 's':
                        ser.write(b's')
                        print("\n>>> Sent START command to ESP32 >>>")
                        # Open CSV file
                        if csv_file is None:
                            csv_file = open(OUTPUT_FILE, 'w', newline='')
                            csv_writer = csv.writer(csv_file)
                            header_written = False
                        collecting = True
                    elif cmd == 'x':
                        ser.write(b'x')
                        print(">>> Sent STOP command to ESP32 >>>\n")
                        collecting = False
                    elif cmd == 'q':
                        print("\nQuitting...")
                        break
            else:
                # For Linux/Mac
                if select.select([sys.stdin], [], [], 0)[0]:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd == 's':
                        ser.write(b's')
                        print("\n>>> Sent START command to ESP32 >>>")
                        if csv_file is None:
                            csv_file = open(OUTPUT_FILE, 'w', newline='')
                            csv_writer = csv.writer(csv_file)
                            header_written = False
                        collecting = True
                    elif cmd == 'x':
                        ser.write(b'x')
                        print(">>> Sent STOP command to ESP32 >>>\n")
                        collecting = False
                    elif cmd == 'q':
                        print("\nQuitting...")
                        break
            
            # Read from serial
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        print(line)  # Display on console
                        
                        # Write to CSV if collecting
                        if collecting and csv_writer:
                            # Check if it's the header line
                            if 'Time(ms)' in line and not header_written:
                                csv_writer.writerow(line.split(','))
                                header_written = True
                            # Check if it's data (starts with a number)
                            elif line and line[0].isdigit():
                                csv_writer.writerow(line.split(','))
                                csv_file.flush()  # Ensure data is written
                except UnicodeDecodeError:
                    pass  # Ignore decode errors
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse
        
        # Cleanup
        if csv_file:
            csv_file.close()
            print(f"\nData saved to: {OUTPUT_FILE}")
        ser.close()
        print("Connection closed.")
        
    except serial.SerialException as e:
        print(f"Error: Could not open {SERIAL_PORT}")
        print(f"Details: {e}")
        print("\nMake sure:")
        print("1. ESP32 is connected")
        print("2. No other program (Arduino IDE, Serial Monitor) is using the port")
        print("3. Correct COM port is specified")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if csv_file:
            csv_file.close()
            print(f"Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
