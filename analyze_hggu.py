import pandas as pd

df = pd.read_csv('HGGU_dual_dataset_20251212_044536.csv')

print('Total rows:', len(df))
print('\nTrials:', sorted(df['trial_id'].unique()))

for trial in [1, 2]:
    t = df[df['trial_id'] == trial]
    if len(t) == 0:
        continue
    
    print(f'\n{"="*70}')
    print(f'Trial {trial}:')
    print(f'  Samples: {len(t)}')
    
    # RIGHT glove analysis
    print(f'\n  RIGHT GLOVE (COM6):')
    print(f'    Accel - ax: {t["ax_right"].min():.4f} to {t["ax_right"].max():.4f}')
    print(f'           ay: {t["ay_right"].min():.4f} to {t["ay_right"].max():.4f}')
    print(f'           az: {t["az_right"].min():.4f} to {t["az_right"].max():.4f}')
    print(f'    Gyro  - gx: {t["gx_right"].min():.3f} to {t["gx_right"].max():.3f}')
    print(f'           gy: {t["gy_right"].min():.3f} to {t["gy_right"].max():.3f}')
    print(f'           gz: {t["gz_right"].min():.3f} to {t["gz_right"].max():.3f}')
    
    # Check if RIGHT frozen
    ax_r_range = t["ax_right"].max() - t["ax_right"].min()
    gx_r_range = t["gx_right"].max() - t["gx_right"].min()
    if ax_r_range < 0.001 and gx_r_range < 0.01:
        print('    ❌ RIGHT GLOVE FROZEN!')
    
    # LEFT glove analysis
    print(f'\n  LEFT GLOVE (COM7):')
    print(f'    Accel - ax: {t["ax_left"].min():.4f} to {t["ax_left"].max():.4f}')
    print(f'           ay: {t["ay_left"].min():.4f} to {t["ay_left"].max():.4f}')
    print(f'           az: {t["az_left"].min():.4f} to {t["az_left"].max():.4f}')
    print(f'    Gyro  - gx: {t["gx_left"].min():.3f} to {t["gx_left"].max():.3f}')
    print(f'           gy: {t["gy_left"].min():.3f} to {t["gy_left"].max():.3f}')
    print(f'           gz: {t["gz_left"].min():.3f} to {t["gz_left"].max():.3f}')
    
    # Check if LEFT frozen
    ax_l_range = t["ax_left"].max() - t["ax_left"].min()
    gx_l_range = t["gx_left"].max() - t["gx_left"].min()
    if ax_l_range < 0.001 and gx_l_range < 0.01:
        print('    ❌ LEFT GLOVE FROZEN!')
    else:
        print('    ✅ LEFT GLOVE WORKING!')

print('\n' + '='*70)
print('Expected behavior:')
print('  Trial 1: No movement (static) - small variations expected')
print('  Trial 2: Random both hand movement - large gyro/accel variations')
