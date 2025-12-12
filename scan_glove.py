"""
Find LEFT glove IP on Roshan hotspot
"""
import socket
import threading

print("Scanning for LEFT glove on port 8080...")

found_ips = []

def check_ip(ip):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, 8080))
        sock.close()
        if result == 0:
            print(f"✅ FOUND: {ip}:8080")
            found_ips.append(ip)
    except:
        pass

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
local_ip = s.getsockname()[0]
s.close()

subnet = '.'.join(local_ip.split('.')[:3])
print(f"Your IP: {local_ip}")
print(f"Scanning: {subnet}.x\n")

threads = []
for i in range(1, 255):
    ip = f"{subnet}.{i}"
    if ip != local_ip:
        t = threading.Thread(target=check_ip, args=(ip,))
        t.start()
        threads.append(t)

for t in threads:
    t.join()

print(f"\n{'='*60}")
if found_ips:
    print(f"Found {len(found_ips)} device(s):")
    for ip in found_ips:
        print(f"  {ip}")
else:
    print("No devices found on port 8080")
