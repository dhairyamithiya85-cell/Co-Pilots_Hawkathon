from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

ea = EventAccumulator('logs/')
ea.Reload()
tags = ea.Tags()['scalars']
print('Tags found:', tags)

data = {}
for tag in tags:
    events = ea.Scalars(tag)
    data[tag] = [(e.step, e.value) for e in events]
    print(f'\n{tag}:')
    for e in events:
        print(f'  step={e.step}, value={e.value:.4f}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax = axes[0]
for tag in tags:
    if 'loss' in tag.lower():
        steps, vals = zip(*data[tag])
        ax.plot(steps, vals, label=tag)
ax.set_title('Loss vs Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

# IoU plot
ax = axes[1]
for tag in tags:
    if 'iou' in tag.lower() or 'miou' in tag.lower():
        steps, vals = zip(*data[tag])
        ax.plot(steps, vals, label=tag)
ax.set_title('mIoU vs Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('mIoU')
ax.legend()
ax.grid(True)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/training_graphs.png', dpi=150)
print('\nSaved graph to outputs/training_graphs.png')
