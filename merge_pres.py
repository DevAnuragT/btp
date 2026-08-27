with open('/home/samyak/code/Labs/BTP_2/presentation.tex', 'r') as f:
    local_lines = f.readlines()
with open('/home/samyak/code/Labs/BTP_2/remote_presentation.tex', 'r') as f:
    remote_lines = f.readlines()

local_idx = next(i for i, line in enumerate(local_lines) if 'SLIDE 10:' in line)
# Adjust to keep the previous line if it was a divider, wait, local_idx is the line '% SLIDE 10...'
# Actually, the line before it is '% =========================================='
local_idx -= 1

remote_idx = next(i for i, line in enumerate(remote_lines) if 'SLIDE 10:' in line)
remote_idx -= 1

merged = local_lines[:local_idx] + remote_lines[remote_idx:]
with open('/home/samyak/code/Labs/BTP_2/presentation.tex', 'w') as f:
    f.writelines(merged)
