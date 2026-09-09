"""Print body names from OptiTrack without importing GMR __init__."""
import sys, time, threading, importlib

# Directly load the NatNetClient module bypassing GMR's __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "NatNetClient",
    "/home/hank/GMR/general_motion_retargeting/optitrack_vendor/NatNetClient.py"
)
nclib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nclib)

setup_optitrack = nclib.setup_optitrack

client = setup_optitrack("192.168.3.103", "192.168.3.137", use_multicast=True)
t = threading.Thread(target=client.run, daemon=True)
t.start()
time.sleep(3)

print(f"Connected: {client.connected()}")
for i in range(5):
    frame = client.get_frame(timeout=3.0)
    if frame:
        print(f"OptiTrack bodies ({len(frame)}):")
        for name in sorted(frame.keys()):
            print(f"  '{name}'")
        break
    print(f"Wait {i}...")
client.shutdown()
