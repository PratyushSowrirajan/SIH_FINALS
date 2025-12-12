import pandas as pd

df = pd.read_csv('asdf_dataset_20251212_062225.csv')

print('Total rows:', len(df))
print('\nTrials:', sorted(df['trial_id'].unique()))

for trial in sorted(df['trial_id'].unique()):
    t = df[df['trial_id'] == trial]
    
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
        print('  ❌ ACCELEROMETER FROZEN!')
    else:
        print('  ✅ Accelerometer working')
        
    if gx_range < 0.01 and gy_range < 0.01 and gz_range < 0.01:
        print('  ❌ GYROSCOPE FROZEN!')
    else:
        print('  ✅ Gyroscope working')

print('\n' + '='*60)
