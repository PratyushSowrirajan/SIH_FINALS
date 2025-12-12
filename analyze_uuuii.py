import pandas as pd

df = pd.read_csv('UUUII_dataset_20251212_053504.csv')

print('Total rows:', len(df))
print('\nTrials:', sorted(df['trial_id'].unique()))

for trial in [1, 2, 3, 4, 5]:
    t = df[df['trial_id'] == trial]
    if len(t) == 0:
        continue
    
    print(f'\n{"="*60}')
    print(f'Trial {trial}:')
    print(f'  Samples: {len(t)}')
    print(f'  Accel - ax: {t["ax"].min():.3f} to {t["ax"].max():.3f}')
    print(f'         ay: {t["ay"].min():.3f} to {t["ay"].max():.3f}')
    print(f'         az: {t["az"].min():.3f} to {t["az"].max():.3f}')
    print(f'  Gyro  - gx: {t["gx"].min():.3f} to {t["gx"].max():.3f}')
    print(f'         gy: {t["gy"].min():.3f} to {t["gy"].max():.3f}')
    print(f'         gz: {t["gz"].min():.3f} to {t["gz"].max():.3f}')
    
    # Check if frozen
    ax_range = t["ax"].max() - t["ax"].min()
    ay_range = t["ay"].max() - t["ay"].min()
    az_range = t["az"].max() - t["az"].min()
    gx_range = t["gx"].max() - t["gx"].min()
    gy_range = t["gy"].max() - t["gy"].min()
    gz_range = t["gz"].max() - t["gz"].min()
    
    if ax_range < 0.001 and ay_range < 0.001 and az_range < 0.001:
        print('  ⚠️  ACCELEROMETER FROZEN!')
    if gx_range < 0.01 and gy_range < 0.01 and gz_range < 0.01:
        print('  ⚠️  GYROSCOPE FROZEN!')

print('\n' + '='*60)
print('Expected behavior:')
print('  Trials 1,2,5: Small variations (static)')
print('  Trial 3: az should be negative (upside down)')
print('  Trial 4: Large gyro/accel variations (shake)')
