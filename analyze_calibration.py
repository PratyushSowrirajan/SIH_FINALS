"""
Analyze calibration quality from after_cal.csv
"""

import csv
import statistics

print("=" * 60)
print("CALIBRATION QUALITY ANALYSIS")
print("=" * 60)

# Read data
ax_vals = []
ay_vals = []
az_vals = []
gx_vals = []
gy_vals = []
gz_vals = []

with open('after_cal.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ax_vals.append(float(row['ax_g']))
        ay_vals.append(float(row['ay_g']))
        az_vals.append(float(row['az_g']))
        gx_vals.append(float(row['gx_dps']))
        gy_vals.append(float(row['gy_dps']))
        gz_vals.append(float(row['gz_dps']))

print(f"\n📊 Analyzed {len(ax_vals)} samples\n")

print("GYROSCOPE (should be ~0 when still):")
print(f"  GX: mean={statistics.mean(gx_vals):+.3f}, std={statistics.stdev(gx_vals):.3f} deg/s")
print(f"  GY: mean={statistics.mean(gy_vals):+.3f}, std={statistics.stdev(gy_vals):.3f} deg/s")
print(f"  GZ: mean={statistics.mean(gz_vals):+.3f}, std={statistics.stdev(gz_vals):.3f} deg/s")

print(f"\nACCELEROMETER (one axis ≈ ±1g, others ≈ 0):")
print(f"  AX: mean={statistics.mean(ax_vals):+.4f}, std={statistics.stdev(ax_vals):.4f} g")
print(f"  AY: mean={statistics.mean(ay_vals):+.4f}, std={statistics.stdev(ay_vals):.4f} g")
print(f"  AZ: mean={statistics.mean(az_vals):+.4f}, std={statistics.stdev(az_vals):.4f} g")

print("\n" + "=" * 60)
print("QUALITY ASSESSMENT:")
print("=" * 60)

# Check gyro drift
gx_drift = abs(statistics.mean(gx_vals))
gy_drift = abs(statistics.mean(gy_vals))
gz_drift = abs(statistics.mean(gz_vals))
max_gyro_drift = max(gx_drift, gy_drift, gz_drift)

if max_gyro_drift < 0.5:
    print("✅ GYRO: EXCELLENT (drift < 0.5 deg/s)")
elif max_gyro_drift < 1.0:
    print("✅ GYRO: GOOD (drift < 1.0 deg/s)")
elif max_gyro_drift < 2.0:
    print("⚠️  GYRO: ACCEPTABLE (drift < 2.0 deg/s)")
else:
    print("❌ GYRO: NEEDS RECALIBRATION (drift > 2.0 deg/s)")

# Check accel
ax_mean = statistics.mean(ax_vals)
ay_mean = statistics.mean(ay_vals)
az_mean = statistics.mean(az_vals)

gravity_magnitude = (ax_mean**2 + ay_mean**2 + az_mean**2)**0.5
print(f"\n✅ ACCEL: Total gravity = {gravity_magnitude:.3f}g (should be ~1.0g)")

if 0.95 < gravity_magnitude < 1.05:
    print("✅ ACCEL: EXCELLENT calibration!")
elif 0.9 < gravity_magnitude < 1.1:
    print("✅ ACCEL: GOOD calibration!")
else:
    print("⚠️  ACCEL: Consider recalibration")

print("\n" + "=" * 60)
